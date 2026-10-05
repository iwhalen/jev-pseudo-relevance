# /// script
# requires-python = ">=3.14"
# dependencies = ["datasets>=5.0.1", "huggingface-hub>=1.0", "httpx>=0.28", "tqdm>=4.70.1"]
# ///

"""Build and publish the official TREC DL 2023 judged passage subset."""

import argparse
import gzip
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path
from typing import BinaryIO

import httpx
from datasets import Dataset, DatasetDict, Features, Value, load_dataset
from huggingface_hub import HfApi
from tqdm import tqdm

QUERY_URL = "https://msmarco.z22.web.core.windows.net/msmarcoranking/2023_queries.tsv"
QREL_URL = "https://trec.nist.gov/data/deep/2023.qrels.pass.withDupes.txt"
CORPUS_URL = (
    "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco_v2_passage.tar"
)
PID_PATTERN = re.compile(r"msmarco_passage_(\d+)_(\d+)\Z")


class ResponseReader(io.RawIOBase):
    """Adapt an HTTP byte iterator to the file interface tarfile requires."""

    def __init__(self, chunks, progress):
        self.chunks = iter(chunks)
        self.pending = bytearray()
        self.progress = progress

    def readable(self):
        return True

    def readinto(self, buffer):
        while not self.pending:
            try:
                chunk = next(self.chunks)
            except StopIteration:
                return 0
            self.progress.update(len(chunk))
            self.pending.extend(chunk)
        size = min(len(buffer), len(self.pending))
        buffer[:size] = self.pending[:size]
        del self.pending[:size]
        return size


def download_metadata(client, url: str, destination: Path) -> bytes:
    if destination.exists():
        return destination.read_bytes()
    response = client.get(url)
    response.raise_for_status()
    content = response.content
    temporary = destination.with_suffix(".partial")
    temporary.write_bytes(content)
    temporary.replace(destination)
    return content


def parse_metadata(query_data: bytes, qrel_data: bytes):
    queries = {}
    for number, line in enumerate(query_data.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split("\t", 1)
        if len(fields) != 2 or not fields[0] or not fields[1]:
            raise ValueError(f"Malformed query at line {number}")
        qid, text = fields
        if qid in queries and queries[qid] != text:
            raise ValueError(f"Conflicting query {qid}")
        queries[qid] = text
    qrels = []
    pairs = set()
    for number, line in enumerate(qrel_data.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 4:
            raise ValueError(f"Malformed qrel at line {number}")
        qid, _, pid, grade = fields
        relevance = int(grade)
        if relevance not in range(4) or not PID_PATTERN.fullmatch(pid):
            raise ValueError(f"Invalid qrel at line {number}")
        if (qid, pid) in pairs:
            raise ValueError(f"Duplicate judgment: {qid}, {pid}")
        if qid not in queries:
            raise ValueError(f"Missing query: {qid}")
        pairs.add((qid, pid))
        qrels.append({"query_id": qid, "passage_id": pid, "relevance": relevance})
    if not qrels:
        raise ValueError("No judgments found")
    judged_queries = {row["query_id"] for row in qrels}
    return (
        [{"query_id": qid, "text": queries[qid]} for qid in sorted(judged_queries)],
        sorted(qrels, key=lambda row: (row["query_id"], row["passage_id"])),
    )


def extract_passages(stream: BinaryIO, required: set[str], progress=None):
    requests = {}
    for pid in required:
        match = PID_PATTERN.fullmatch(pid)
        if match is None:
            raise ValueError(f"Invalid passage ID: {pid}")
        shard, offset = match.groups()
        requests.setdefault(f"msmarco_passage_{shard}", []).append((int(offset), pid))
    passages = {}
    with tarfile.open(fileobj=stream, mode="r|") as archive:
        for member in archive:
            name = Path(member.name).name.removesuffix(".gz")
            if not member.isfile() or name not in requests:
                continue
            compressed = archive.extractfile(member)
            if compressed is None:
                raise ValueError(f"Cannot read shard {member.name}")
            with compressed, gzip.GzipFile(fileobj=compressed) as shard:
                for offset, pid in sorted(requests.pop(name)):
                    # Offsets refer to UTF-8 bytes in the uncompressed JSONL.
                    shard.seek(offset)
                    record = json.loads(shard.readline())
                    if record.get("pid") != pid:
                        raise ValueError(
                            f"Passage ID mismatch at offset {offset}: {pid}"
                        )
                    text, docid, spans = (
                        record[key] for key in ("passage", "docid", "spans")
                    )
                    if not all(
                        isinstance(value, str) for value in (text, docid, spans)
                    ):
                        raise ValueError(f"Invalid passage fields: {pid}")
                    passages[pid] = {
                        "passage_id": pid,
                        "text": text,
                        "document_id": docid,
                        "spans": spans,
                    }
                    if progress is not None:
                        progress.update(1)
            if len(passages) == len(required):
                break
    missing = required - passages.keys()
    if missing:
        raise ValueError(
            f"Missing {len(missing)} passages; examples: {sorted(missing)[:10]}"
        )
    return [passages[pid] for pid in sorted(passages)]


def validate_tables(queries, passages, qrels):
    query_ids = [row["query_id"] for row in queries]
    passage_ids = [row["passage_id"] for row in passages]
    pairs = [(row["query_id"], row["passage_id"]) for row in qrels]
    if len(set(query_ids)) != len(query_ids) or len(set(passage_ids)) != len(
        passage_ids
    ):
        raise ValueError("Duplicate query or passage IDs")
    if len(set(pairs)) != len(pairs):
        raise ValueError("Duplicate judgments")
    if set(query_ids) != {qid for qid, _ in pairs}:
        raise ValueError("Query IDs differ from judged query IDs")
    if set(passage_ids) != {pid for _, pid in pairs}:
        raise ValueError("Passage IDs differ from judged passage IDs")
    if any(row["relevance"] not in range(4) for row in qrels):
        raise ValueError("Invalid relevance grade")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "repo_id", help="Destination Hugging Face dataset repo: owner/name"
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path(".cache/trec-dl-2023-judged")
    )
    parser.add_argument(
        "--private", action="store_true", help="Create a private repository"
    )
    args = parser.parse_args()
    if len(args.repo_id.split("/")) != 2 or not all(args.repo_id.split("/")):
        parser.error("repo_id must have the form owner/name")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    with httpx.Client(
        follow_redirects=True,
        timeout=httpx.Timeout(120, connect=30),
        transport=httpx.HTTPTransport(retries=3),
        headers={"X-Ms-Version": "2019-12-12", "Accept-Encoding": "identity"},
    ) as client:
        query_data = download_metadata(
            client, QUERY_URL, args.cache_dir / "2023_queries.tsv"
        )
        qrel_data = download_metadata(
            client, QREL_URL, args.cache_dir / "2023.qrels.pass.withDupes.txt"
        )
        queries, qrels = parse_metadata(query_data, qrel_data)
        required = {row["passage_id"] for row in qrels}
        fingerprint = hashlib.sha256(qrel_data).hexdigest()
        subset_path = args.cache_dir / f"passages-{fingerprint}.json"
        if subset_path.exists():
            passages = json.loads(subset_path.read_text(encoding="utf-8"))
        else:
            with client.stream("GET", CORPUS_URL) as response:
                response.raise_for_status()
                total = response.headers.get("Content-Length")
                with (
                    tqdm(
                        total=int(total) if total else None,
                        unit="B",
                        unit_scale=True,
                        desc="Corpus transfer",
                    ) as transfer,
                    tqdm(total=len(required), desc="Judged passages") as extracted,
                    io.BufferedReader(
                        ResponseReader(response.iter_bytes(), transfer)
                    ) as stream,
                ):
                    passages = extract_passages(stream, required, extracted)
            temporary = subset_path.with_suffix(".partial")
            temporary.write_text(
                json.dumps(passages, ensure_ascii=False), encoding="utf-8"
            )
            temporary.replace(subset_path)
    validate_tables(queries, passages, qrels)
    schemas = {
        "queries": Features({"query_id": Value("string"), "text": Value("string")}),
        "passages": Features(
            {
                key: Value("string")
                for key in ("passage_id", "text", "document_id", "spans")
            }
        ),
        "qrels": Features(
            {
                "query_id": Value("string"),
                "passage_id": Value("string"),
                "relevance": Value("int32"),
            }
        ),
    }
    tables = DatasetDict(
        {
            name: Dataset.from_list(rows, features=schemas[name])
            for name, rows in (
                ("queries", queries),
                ("passages", passages),
                ("qrels", qrels),
            )
        }
    )
    local_path = args.cache_dir / f"dataset-{fingerprint}"
    if not local_path.exists():
        tables.save_to_disk(str(local_path))
    api = HfApi()
    api.create_repo(
        args.repo_id, repo_type="dataset", private=args.private, exist_ok=True
    )
    for name, table in tables.items():
        DatasetDict({"test": table}).push_to_hub(
            args.repo_id,
            config_name=name,
            private=args.private,
        )
    revision = api.dataset_info(args.repo_id).sha

    for name, table in tables.items():
        hosted = load_dataset(args.repo_id, name, split="test", revision=revision)
        if hosted.features != table.features or hosted.to_list() != table.to_list():
            raise ValueError(f"Hosted {name} does not match local data at {revision}")
    print(
        f"Validated upload: https://huggingface.co/datasets/{args.repo_id}/tree/{revision}"
    )


if __name__ == "__main__":
    main()
