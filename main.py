"""Judge TREC DL 2023 passages and compare assessment and system agreement."""

import argparse
import asyncio
import gzip
import hashlib
import json
import math
import random
import re
import time
from pathlib import Path
from typing import Literal
from urllib.request import urlopen

import ir_measures
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from datasets import Dataset, load_dataset
from openai import (
    APIConnectionError,
    APIStatusError,
    AsyncOpenAI,
    BadRequestError,
    RateLimitError,
)
from pydantic import BaseModel, ConfigDict
from scipy.stats import kendalltau
from tqdm import tqdm
from typesafe_sdk import (
    AsyncTypeSafeClient,
    RetryPolicy,
    Score,
    TypeSafeBadRequestError,
)

plt.style.use("dark_background")
DATA_DIR = Path(__file__).parent / "data"
DATASET = "iwhalen/trec-dl-2023-judged"
# Conservative character estimate for the 32k state-plus-question token limit.
JEV_REQUEST_CHAR_BUDGET = 120_000
RUN_BASE = (
    "https://raw.githubusercontent.com/rahmanidashti/"
    "SyntheticTestCollections/main/dl-2023-runs/"
)
RUN_NAMES = [
    "agg-cocondenser",
    "bm25_splades",
    "cip_run_1",
    "cip_run_2",
    "cip_run_3",
    "cip_run_4",
    "cip_run_5",
    "cip_run_6",
    "cip_run_7",
    "naverloo-frgpt4",
    "naverloo-rgpt4",
    "naverloo_bm25_RR",
    "naverloo_bm25_splades_RR",
    "naverloo_fs",
    "naverloo_fs_RR",
    "naverloo_fs_RR_duo",
    "slim-pp-0shot-uw",
    "splade_pp_ensemble_distil",
    "splade_pp_self_distil",
    "uogtr_b_grf_e",
    "uogtr_b_grf_e_gb",
    "uogtr_be",
    "uogtr_be_gb",
    "uogtr_dph",
    "uogtr_dph_bo1",
    "uogtr_qr_be",
    "uogtr_qr_be_gb",
    "uogtr_s",
    "uogtr_se",
    "uogtr_se_gb",
    "uot-yahoo_LLMs-blender",
    "uot-yahoo_rankgpt35",
    "uot-yahoo_rankgpt4",
    "WatS-Augmented-BM25",
    "WatS-LLM-Rerank",
]
INSTRUCTIONS = (
    "Rate the passage's relevance to the query using the criteria. "
    "Use only information in the passage."
)
CRITERIA = [
    "Irrelevant: The passage has nothing to do with the query.",
    "Related: The passage seems related to the query but does not answer it.",
    (
        "Highly relevant: The passage has some answer for the query, but the answer "
        "may be a bit unclear, or hidden amongst extraneous information."
    ),
    "Perfectly relevant: The passage is dedicated to the query and contains the exact answer.",
]

# Token costs per million.
TOKEN_COSTS = {
    "jev": {
        "input": 0.042,
        "output": 0.0,
    },
    "gpt-6-luna": {
        "input": 0.1,
        "output": 0.5,
    },
}


def save_json(path: Path, value) -> None:
    """Replace JSON atomically, rejecting nonstandard NaN values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_run(path: Path) -> dict:
    """Accept zero- or one-based ranks; trec_eval orders by score, not rank."""
    opener = gzip.open if ".gz" in path.suffixes else open
    run = {}
    with opener(path, "rt", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 6:
                raise ValueError(f"{path}:{number}: expected six TREC columns")
            qid, _, pid, rank, score, _ = fields
            if not pid.startswith("msmarco_passage_") or int(rank) < 0:
                raise ValueError(f"{path}:{number}: invalid passage or rank")
            score = float(score)
            documents = run.setdefault(qid, {})
            if not math.isfinite(score) or pid in documents:
                raise ValueError(f"{path}:{number}: duplicate passage or invalid score")
            documents[pid] = score
    if not run:
        raise ValueError(f"Empty run: {path}")
    return run


def download_runs(runs_dir: Path = DATA_DIR / "runs") -> list[Path]:
    """Cache GitHub passage runs; existing files are never downloaded."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name in tqdm(RUN_NAMES, desc="GitHub runs"):
        path = runs_dir / f"{name}.run"
        url = RUN_BASE + name
        if not path.exists():
            temporary = path.with_suffix(".run.partial")
            try:
                with urlopen(url, timeout=120) as response:
                    with temporary.open("wb") as output:
                        while chunk := response.read(1024 * 1024):
                            output.write(chunk)
                load_run(temporary)
                temporary.replace(path)
            except Exception as error:
                temporary.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Cannot download {url}. Populate data/runs/ and use --local-runs."
                ) from error
        load_run(path)
        manifest.append(
            {
                "file": path.name,
                "url": url,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    save_json(runs_dir / "manifest.json", manifest)
    return [runs_dir / entry["file"] for entry in manifest]


def validate_judgments(rows: list[dict]) -> None:
    pairs = set()
    for row in rows:
        pair = (row["query_id"], row["passage_id"])
        if pair in pairs:
            raise ValueError(f"Duplicate judgment: {pair}")
        pairs.add(pair)
        for field in ("human_relevance", "predicted_relevance"):
            if type(row[field]) is not int or row[field] not in range(4):
                raise ValueError(f"Invalid {field}: {pair}")


def judgment_dataset(rows: list[dict]) -> Dataset:
    """Preserve optional fields when resuming older, less detailed checkpoints."""
    columns = dict.fromkeys(key for row in rows for key in row)
    return Dataset.from_dict({key: [row.get(key) for row in rows] for key in columns})


class RelevanceJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    grade: Literal[0, 1, 2, 3]


DEFAULT_MODELS = {"jev": "jev-latest", "openai": "gpt-6-luna"}
REASONING_EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")
OPENAI_REQUESTS_PER_MINUTE = 450
OPENAI_TOKENS_PER_MINUTE = 180_000


class OpenAIRateLimiter:
    """Pace all workers together, including retries, without accumulating bursts."""

    def __init__(self, requests_per_minute, tokens_per_minute):
        if requests_per_minute <= 0 or tokens_per_minute <= 0:
            raise ValueError("OpenAI rate limits must be positive")
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.next_request = 0.0
        self.lock = asyncio.Lock()

    async def acquire(self, estimated_tokens):
        while True:
            async with self.lock:
                now = time.monotonic()
                delay = self.next_request - now
                if delay <= 0:
                    self.next_request = now + max(
                        60 / self.requests_per_minute,
                        60 * estimated_tokens / self.tokens_per_minute,
                    )
                    return
            # Recheck after waking: another worker or a 429 may extend the wait.
            await asyncio.sleep(min(delay, 60))

    async def cooldown(self, seconds):
        async with self.lock:
            self.next_request = max(self.next_request, time.monotonic() + seconds)


def openai_retry_delay(error, attempt):
    """Respect server retry hints, with exponential backoff and jitter."""
    delay = min(60, 2**attempt) + random.uniform(0, 0.5)
    headers = error.response.headers if isinstance(error, APIStatusError) else {}
    for name, scale in (("retry-after-ms", 0.001), ("retry-after", 1)):
        try:
            hint = float(headers[name]) * scale
            if math.isfinite(hint):
                delay = max(delay, hint)
        except KeyError, ValueError:
            pass
    match = re.search(r"try again in ([\d.]+)(ms|s)", str(error), re.IGNORECASE)
    if match:
        delay = max(delay, float(match[1]) * (0.001 if match[2].lower() == "ms" else 1))
    return delay


def openai_instructions():
    return (
        INSTRUCTIONS
        + "\n"
        + "\n".join(
            f"[{grade}] {criterion}" for grade, criterion in enumerate(CRITERIA)
        )
    )


def truncate_passage(query, passage):
    """Use the same conservative initial passage prefix for both providers."""
    overhead = len(
        json.dumps(
            {
                "query": query,
                "passage": "",
                "instructions": INSTRUCTIONS,
                "criteria": CRITERIA,
            },
            ensure_ascii=False,
        )
    )
    return passage[: max(1, JEV_REQUEST_CHAR_BUDGET - overhead - 2_000)]


def is_context_error(error):
    if isinstance(error, TypeSafeBadRequestError):
        return "max_tokens_exceeded" in str(error)
    return (
        isinstance(error, BadRequestError) and error.code == "context_length_exceeded"
    )


def response_usage(response):
    usage = response.usage
    return {
        "input_tokens": usage.input_tokens if usage is not None else None,
        "output_tokens": usage.output_tokens if usage is not None else None,
    }


async def request_jev(client, query, passage):
    response = await client.system_one(
        state={"query": query, "passage": passage},
        questions={"relevance": Score(instructions=INSTRUCTIONS, criteria=CRITERIA)},
    )
    answer = response.scores["relevance"]
    probabilities = dict(answer.probabilities)
    return {
        "predicted_relevance": max(range(4), key=probabilities.get),
        "score": float(answer.score),
        "confidence": float(answer.confidence),
        "probabilities": {str(grade): value for grade, value in probabilities.items()},
        "usage": response_usage(response),
        "request_id": response.request_id,
        "model": response.model,
    }


async def request_openai(client, query, passage, *, model, reasoning_effort):
    response = await client.responses.parse(
        model=model,
        reasoning={"effort": reasoning_effort},
        instructions=openai_instructions(),
        input=json.dumps({"query": query, "passage": passage}, ensure_ascii=False),
        text_format=RelevanceJudgment,
        store=False,
    )
    if response.status != "completed" or response.output_parsed is None:
        raise ValueError(
            "OpenAI judgment incomplete, refused, or missing structured output"
        )
    return {
        "predicted_relevance": response.output_parsed.grade,
        "usage": response_usage(response),
        "request_id": response._request_id,
        "model": response.model,
    }


async def judge_async(
    queries,
    passages,
    qrels,
    *,
    output_dir,
    model,
    concurrency,
    method,
    client_factory,
    request,
    configuration,
):
    """Judge independently with bounded workers and durable per-pair checkpoints."""
    if concurrency < 1:
        raise ValueError("Concurrency must be positive")
    output_dir = Path(output_dir)
    query_text = {row["query_id"]: row["text"] for row in queries}
    passage_text = {row["passage_id"]: row["text"] for row in passages}
    pairs = [(row["query_id"], row["passage_id"]) for row in qrels]
    if len(set(pairs)) != len(pairs):
        raise ValueError("Duplicate reference query-passage pairs")
    for qid, pid in pairs:
        if qid not in query_text or pid not in passage_text:
            raise ValueError(f"Missing query or passage: {qid}, {pid}")
    signature = hashlib.sha256(
        json.dumps(
            {
                "model": model,
                "instructions": INSTRUCTIONS,
                "criteria": CRITERIA,
                "inputs": [
                    (qid, pid, query_text[qid], passage_text[pid]) for qid, pid in pairs
                ],
                **configuration,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "experiment.json"
    if metadata_path.exists():
        if json.loads(metadata_path.read_text())[
            "signature"
        ] != signature and read_jsonl(output_dir / "judgments.jsonl"):
            raise ValueError("Cached experiment differs; use a new --data-dir")
    elif (output_dir / "judgments.jsonl").exists():
        raise ValueError("Judgment checkpoint has no experiment metadata")
    save_json(
        metadata_path,
        {
            "signature": signature,
            "dataset": DATASET,
            "model": model,
            "instructions": INSTRUCTIONS,
            "criteria": CRITERIA,
            "concurrency": concurrency,
            **configuration,
        },
    )
    cached = read_jsonl(output_dir / "judgments.jsonl")
    validate_judgments(cached)
    completed = {(r["query_id"], r["passage_id"]): r for r in cached}
    if not completed.keys() <= set(pairs):
        raise ValueError("Checkpoint contains unexpected pairs")
    pending = iter(
        row for row in qrels if (row["query_id"], row["passage_id"]) not in completed
    )
    failures = []
    with tqdm(
        total=len(pairs), initial=len(completed), desc=f"{method} judgments"
    ) as progress:
        if len(completed) < len(pairs):
            async with client_factory() as client:
                with (output_dir / "judgments.jsonl").open(
                    "a", encoding="utf-8"
                ) as checkpoint:

                    async def worker():
                        for row in pending:
                            qid, pid = row["query_id"], row["passage_id"]
                            start = time.perf_counter()
                            try:
                                passage = truncate_passage(
                                    query_text[qid], passage_text[pid]
                                )
                                length_retries = 0
                                while True:
                                    try:
                                        answer = await request(
                                            client, query_text[qid], passage
                                        )
                                        break
                                    except Exception as error:
                                        if (
                                            not is_context_error(error)
                                            or len(passage) <= 1
                                        ):
                                            raise
                                        passage = passage[: len(passage) // 2]
                                        length_retries += 1
                                result = {
                                    "query_id": qid,
                                    "passage_id": pid,
                                    "human_relevance": int(row["relevance"]),
                                    "passage_original_chars": len(passage_text[pid]),
                                    "passage_used_chars": len(passage),
                                    "passage_truncated": len(passage)
                                    < len(passage_text[pid]),
                                    "length_retries": length_retries,
                                    **answer,
                                    "latency_seconds": time.perf_counter() - start,
                                }
                                validate_judgments([result])
                                checkpoint.write(
                                    json.dumps(result, allow_nan=False) + "\n"
                                )
                                checkpoint.flush()
                                completed[(qid, pid)] = result
                            except Exception as error:
                                failures.append(
                                    {
                                        "query_id": qid,
                                        "passage_id": pid,
                                        "error": str(error),
                                        "type": type(error).__name__,
                                    }
                                )
                            progress.update(1)

                    await asyncio.gather(*(worker() for _ in range(concurrency)))
    save_json(output_dir / "failures.json", failures)
    if failures:
        raise RuntimeError(
            f"{len(failures)} judgments failed; rerun to resume. See failures.json"
        )
    # Refresh reference labels rather than trusting cached copies.
    return judgment_dataset(
        [
            {**completed[pair], "human_relevance": int(row["relevance"])}
            for pair, row in zip(pairs, qrels, strict=True)
        ]
    )


async def judge_jev_async(queries, passages, qrels, *, output_dir, model, concurrency):
    model = "jev-latest" if model == "jev" else model
    return await judge_async(
        queries,
        passages,
        qrels,
        output_dir=output_dir,
        model=model,
        concurrency=concurrency,
        method="Jev",
        request=request_jev,
        client_factory=lambda: AsyncTypeSafeClient(
            model=model, retry=RetryPolicy(max_retries=4)
        ),
        configuration={"decision": "argmax_probability_lower_grade_on_tie"},
    )


async def judge_openai_async(
    queries,
    passages,
    qrels,
    *,
    output_dir,
    model,
    concurrency,
    reasoning_effort="none",
    requests_per_minute=OPENAI_REQUESTS_PER_MINUTE,
    tokens_per_minute=OPENAI_TOKENS_PER_MINUTE,
):
    if reasoning_effort not in REASONING_EFFORTS:
        raise ValueError(f"Unsupported reasoning effort: {reasoning_effort}")
    limiter = OpenAIRateLimiter(requests_per_minute, tokens_per_minute)
    prompt_overhead = openai_instructions() + json.dumps(
        RelevanceJudgment.model_json_schema()
    )

    async def request(client, query, passage):
        # Conservative byte estimate plus room for framing and the small answer.
        text = prompt_overhead + json.dumps(
            {"query": query, "passage": passage}, ensure_ascii=False
        )
        estimated_tokens = math.ceil(len(text.encode("utf-8")) / 3) + 128
        for attempt in range(9):
            await limiter.acquire(estimated_tokens)
            try:
                return await request_openai(
                    client,
                    query,
                    passage,
                    model=model,
                    reasoning_effort=reasoning_effort,
                )
            except (APIConnectionError, APIStatusError) as error:
                if isinstance(error, RateLimitError):
                    # Billing/quota errors cannot be resolved by throttling.
                    if error.code != "rate_limit_exceeded":
                        raise
                    await limiter.cooldown(openai_retry_delay(error, attempt))
                    if attempt == 8:
                        raise
                elif (
                    isinstance(error, APIConnectionError)
                    or error.status_code in (408, 409)
                    or error.status_code >= 500
                ) and attempt < 4:
                    await limiter.cooldown(openai_retry_delay(error, attempt))
                else:
                    raise

    return await judge_async(
        queries,
        passages,
        qrels,
        output_dir=output_dir,
        model=model,
        concurrency=concurrency,
        method="OpenAI",
        request=request,
        # Every retry must pass through the shared limiter.
        client_factory=lambda: AsyncOpenAI(max_retries=0),
        configuration={
            "decision": "structured_grade",
            "reasoning_effort": reasoning_effort,
            "prompt": openai_instructions(),
            "input_format": "query_passage_json_v1",
            "output_schema": RelevanceJudgment.model_json_schema(),
            "truncation": {
                "character_budget": JEV_REQUEST_CHAR_BUDGET,
                "margin": 2_000,
            },
        },
    )


def judge_jev(
    queries: Dataset,
    passages: Dataset,
    qrels: Dataset,
    *,
    output_dir=DATA_DIR / "jev",
    model="jev-latest",
    concurrency=16,
) -> Dataset:
    return asyncio.run(
        judge_jev_async(
            queries,
            passages,
            qrels,
            output_dir=output_dir,
            model=model,
            concurrency=concurrency,
        )
    )


def judge_openai(
    queries: Dataset,
    passages: Dataset,
    qrels: Dataset,
    *,
    output_dir=DATA_DIR / "openai",
    model="gpt-6-luna",
    concurrency=16,
    reasoning_effort="none",
    requests_per_minute=OPENAI_REQUESTS_PER_MINUTE,
    tokens_per_minute=OPENAI_TOKENS_PER_MINUTE,
) -> Dataset:
    return asyncio.run(
        judge_openai_async(
            queries,
            passages,
            qrels,
            output_dir=output_dir,
            model=model,
            concurrency=concurrency,
            reasoning_effort=reasoning_effort,
            requests_per_minute=requests_per_minute,
            tokens_per_minute=tokens_per_minute,
        )
    )


METHODS = {"jev": judge_jev, "openai": judge_openai}


def agreement_report(rows):
    counts = np.zeros((4, 4), dtype=int)
    for row in rows:
        counts[row["human_relevance"], row["predicted_relevance"]] += 1
    total = int(counts.sum())
    observed = float(np.trace(counts) / total)
    expected = float(counts.sum(axis=0) @ counts.sum(axis=1) / total**2)
    return {
        "labels": [0, 1, 2, 3],
        "rows": "human",
        "columns": "method",
        "counts": counts.tolist(),
        "count": total,
        "agree_count": int(np.trace(counts)),
        "exact_agreement": observed,
        "cohen_kappa": (observed - expected) / (1 - expected) if expected < 1 else None,
        "undefined_reason": "No variation in either assessor"
        if expected == 1
        else None,
        "human_support": counts.sum(axis=1).tolist(),
        "method_support": counts.sum(axis=0).tolist(),
    }


def performance_report(
    judgments: Dataset, method: str, *, run_paths=None, data_dir=DATA_DIR
) -> dict:
    """Compare paired grades and retrieval-system rankings on identical qrels support."""
    rows = judgments.to_list()
    if not rows:
        raise ValueError("No judgments to report")
    validate_judgments(rows)
    output = Path(data_dir) / method
    output.mkdir(parents=True, exist_ok=True)
    agreement = agreement_report(rows)
    counts = np.array(agreement["counts"])
    for normalized in (False, True):
        values = counts.astype(float)
        if normalized:
            values = np.divide(
                values,
                counts.sum(axis=1, keepdims=True),
                out=np.zeros_like(values),
                where=counts.sum(axis=1, keepdims=True) != 0,
            )
        fig, ax = plt.subplots()
        image = ax.imshow(values, cmap="Blues")
        for i in range(4):
            for j in range(4):
                ax.text(
                    j,
                    i,
                    f"{values[i, j]:.1%}" if normalized else str(counts[i, j]),
                    ha="center",
                    va="center",
                    color="orange",
                )
        ax.set(
            xticks=range(4),
            yticks=range(4),
            xlabel=f"{method} grade",
            ylabel="Human grade",
            title=f"Assessment agreement (n={len(rows):,}, κ={agreement['cohen_kappa']:0.4f})",
        )
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(
            output
            / ("agreement_normalized.png" if normalized else "agreement_counts.png"),
            dpi=180,
        )
        plt.close(fig)
    metrics = [ir_measures.nDCG @ 10, ir_measures.nDCG @ 100, ir_measures.AP(rel=2)]
    references = {}
    for source, field in (
        ("human", "human_relevance"),
        (method, "predicted_relevance"),
    ):
        references[source] = [
            ir_measures.Qrel(r["query_id"], r["passage_id"], r[field]) for r in rows
        ]
    scores = {}
    manifest = []
    run_paths = (
        download_runs(Path(data_dir) / "runs") if run_paths is None else run_paths
    )
    if len(run_paths) < 2:
        raise ValueError(
            "At least two passage runs are required for system rank correlation"
        )
    for path in tqdm(run_paths, desc="Evaluate runs"):
        path = Path(path)
        run = load_run(path)
        if path.name in scores:
            raise ValueError(f"Duplicate run filename: {path.name}")
        scores[path.name] = {}
        for source, qrels in references.items():
            per_query = {
                qid: {str(m): 0.0 for m in metrics}
                for qid in {r["query_id"] for r in rows}
            }
            filtered = {qid: docs for qid, docs in run.items() if qid in per_query}
            for result in ir_measures.iter_calc(metrics, qrels, filtered):
                per_query[result.query_id][str(result.measure)] = float(result.value)
            scores[path.name][source] = {
                "aggregate": {
                    str(m): sum(v[str(m)] for v in per_query.values()) / len(per_query)
                    for m in metrics
                },
                "per_query": per_query,
            }
        manifest.append(
            {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        )
    correlations = {}
    for metric in metrics:
        key = str(metric)
        human = [v["human"]["aggregate"][key] for v in scores.values()]
        predicted = [v[method]["aggregate"][key] for v in scores.values()]
        tau, pvalue = kendalltau(human, predicted, variant="b")
        correlations[key] = {
            "tau_b": float(tau) if math.isfinite(tau) else None,
            "p_value": float(pvalue) if math.isfinite(pvalue) else None,
            "system_count": len(scores),
            "undefined_reason": "Constant system scores"
            if not math.isfinite(tau)
            else None,
        }
        fig, ax = plt.subplots()
        ax.scatter(human, predicted)
        low, high = min(human + predicted), max(human + predicted)
        ax.plot([low, high], [low, high], "--", alpha=0.5)
        ax.set(
            xlabel=f"Human {key}",
            ylabel=f"{method} {key}",
            title=f"System score agreement: τ-b={correlations[key]['tau_b']:0.4f} (n={len(scores)})",
        )
        fig.tight_layout()
        fig.savefig(
            output / f"system_scores_{re.sub(r'[^a-zA-Z0-9]', '_', key)}.png", dpi=180
        )
        plt.close(fig)
    recorded_usage = [
        row["usage"]
        for row in rows
        if row.get("usage") is not None
        and row["usage"].get("input_tokens") is not None
        and row["usage"].get("output_tokens") is not None
    ]
    input_tokens = sum(usage["input_tokens"] for usage in recorded_usage)
    output_tokens = sum(usage["output_tokens"] for usage in recorded_usage)
    metadata_path = output / "experiment.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    model = metadata.get("model", rows[0].get("model", DEFAULT_MODELS.get(method)))
    pricing_model = "jev" if method == "jev" else model
    token_costs = TOKEN_COSTS.get(pricing_model)
    input_cost = (
        input_tokens * token_costs["input"] / 1_000_000 if token_costs else None
    )
    output_cost = (
        output_tokens * token_costs["output"] / 1_000_000 if token_costs else None
    )
    report = {
        "method": method,
        "cost": {
            "input": input_cost,
            "output": output_cost,
            "total": input_cost + output_cost if token_costs else None,
            "estimated": True,
            "model": model,
            "undefined_reason": None
            if token_costs
            else f"No token prices configured for {model}",
            "token_costs_per_million": token_costs,
            "complete": token_costs is not None and len(recorded_usage) == len(rows),
        },
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "judgments_with_usage": len(recorded_usage),
            "judgments_without_usage": len(rows) - len(recorded_usage),
        },
        "dataset": DATASET,
        "judgment_count": len(rows),
        "query_count": len({r["query_id"] for r in rows}),
        "agreement": agreement,
        "system_rank_correlations": correlations,
        "runs": manifest,
        "evaluation": {
            "ndcg_gain": "linear (trec_eval)",
            "binary_relevance_threshold": 2,
            "unjudged": "nonrelevant; rankings are not condensed",
            "reference_qrels": "NIST withDupes; includes propagated duplicate labels",
        },
    }
    save_json(output / "run_scores.json", scores)
    save_json(output / "results.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, default="jev")
    parser.add_argument(
        "--model",
        help="Model ID (default: jev-latest for Jev, gpt-6-luna for OpenAI)",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=REASONING_EFFORTS,
        help="OpenAI reasoning effort (default: none)",
    )
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument(
        "--openai-rpm", type=int, help="OpenAI request budget per minute (default: 450)"
    )
    parser.add_argument(
        "--openai-tpm",
        type=int,
        help="OpenAI estimated token budget per minute (default: 180000)",
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Evaluate a complete checkpoint without API calls",
    )
    parser.add_argument(
        "--local-runs",
        action="store_true",
        help="Use cached .gz, .txt, .trec, or .run files without downloading",
    )
    parser.add_argument(
        "--download-only", action="store_true", help="Cache GitHub runs, then exit"
    )
    args = parser.parse_args()
    if args.method != "openai" and args.reasoning_effort is not None:
        parser.error("--reasoning-effort is only supported with --method openai")
    if args.method != "openai" and (
        args.openai_rpm is not None or args.openai_tpm is not None
    ):
        parser.error("--openai-rpm and --openai-tpm require --method openai")
    if any(
        value is not None and value <= 0 for value in (args.openai_rpm, args.openai_tpm)
    ):
        parser.error("OpenAI rate limits must be positive")
    args.model = args.model or DEFAULT_MODELS[args.method]
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    runs_dir = args.data_dir / "runs"
    paths = (
        sorted(
            p
            for p in runs_dir.glob("*")
            if p.suffix in {".gz", ".txt", ".trec", ".run"}
        )
        if args.local_runs
        else download_runs(runs_dir)
    )
    if args.download_only:
        return
    if len(paths) < 2:
        parser.error("At least two passage runs are required")
    for path in paths:
        load_run(path)
    qrels = load_dataset(DATASET, "qrels", split="test")
    output = args.data_dir / args.method
    if args.report_only:
        rows = read_jsonl(output / "judgments.jsonl")
        validate_judgments(rows)
        expected = {(r["query_id"], r["passage_id"]): r["relevance"] for r in qrels}
        if {(r["query_id"], r["passage_id"]) for r in rows} != expected.keys():
            raise ValueError(
                "Report requires a complete checkpoint matching the reference qrels"
            )
        judgments = judgment_dataset(
            [
                {
                    **r,
                    "human_relevance": int(expected[(r["query_id"], r["passage_id"])]),
                }
                for r in rows
            ]
        )
    else:
        queries = load_dataset(DATASET, "queries", split="test")
        passages = load_dataset(DATASET, "passages", split="test")
        judgments = METHODS[args.method](
            queries,
            passages,
            qrels,
            output_dir=output,
            model=args.model,
            concurrency=args.concurrency,
            **(
                {
                    "reasoning_effort": args.reasoning_effort or "none",
                    "requests_per_minute": args.openai_rpm
                    or OPENAI_REQUESTS_PER_MINUTE,
                    "tokens_per_minute": args.openai_tpm or OPENAI_TOKENS_PER_MINUTE,
                }
                if args.method == "openai"
                else {}
            ),
        )
    report = performance_report(
        judgments, args.method, run_paths=paths, data_dir=args.data_dir
    )
    print(
        json.dumps(
            {
                "usage": report["usage"],
                "cost": report["cost"],
                "agreement": report["agreement"],
                "system_rank_correlations": report["system_rank_correlations"],
            },
            indent=2,
        )
    )
    print(f"Reports saved in {output}")


if __name__ == "__main__":
    main()
