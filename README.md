# Jev pseudo-relevance

To run the experiments, first get your `TYPESAFE_API_KEY` set in a `.env` file.

Then run:

```bash
uv run --env-file .env main.py
```

Outputs will be written to `data/`.

The script downloads the 35 TREC DL 2023 passage submissions from the
[SyntheticTestCollections repository](https://github.com/rahmanidashti/SyntheticTestCollections/tree/main/dl-2023-runs)
to `data/runs/`, reusing existing files. Cache runs independently with:

```bash
uv run --env-file .env main.py --download-only
```

Jev judges each query–passage pair independently with 16 concurrent requests.
Use `--concurrency 8` to change this, and `--model <version>` to pin a model.
Completed judgments are checkpointed; rerunning the same command resumes them.
Changed inputs, prompts, model, or OpenAI reasoning effort require a separate `--data-dir`.

To judge with OpenAI, set `OPENAI_API_KEY` in `.env` and run:

```bash
uv run --env-file .env main.py --method openai
```

OpenAI defaults to `gpt-6-luna` with reasoning effort `none`. Override these
with `--model <model>` and `--reasoning-effort low`. Responses are parsed with
a Pydantic structured-output model containing only an integer grade from 0–3.
Both methods use the same rubric and conservative initial passage truncation
limit; context-length errors cause further passage shortening. Truncation is
recorded per judgment. OpenAI outputs are written to `data/openai/`.

OpenAI workers share a throttle of 450 requests/minute and 180,000 estimated
tokens/minute, below the observed 500 RPM / 200,000 TPM limits. Token estimates
include the rubric, query, passage, schema, and a small output allowance.
Override the budgets with `--openai-rpm 300 --openai-tpm 120000` if your account
has lower limits or other processes share the quota. Rate-limit responses pause
all workers using server retry hints and exponential backoff, with up to eight
retries per judgment. Quota/billing errors fail immediately. Rerunning the same
command resumes existing checkpoints; throttle changes do not invalidate them.

Results in `data/jev/` include `judgments.jsonl`, `experiment.json`,
`failures.json`, `results.json`, `run_scores.json`, and agreement/system-score
plots. Jev's four-level Score probabilities are converted to integer grades
using the most probable level (lower grade on ties), preserving the raw score
and probabilities. The rubric distinguishes related but non-answering passages
(grade 1) from passages providing an answer (grades 2–3).

Reporting computes unweighted Cohen's kappa and a human-row/method-column 4×4
count table. Kendall's tau-b compares **system orderings**, using mean nDCG@10,
nDCG@100, and AP under human versus method qrels. nDCG uses trec_eval's linear gains;
AP treats only grades 2–3 as relevant. Unjudged passages stay in the rankings.
Headline reports require all reference judgments to be present. Token totals
and estimated costs use `TOKEN_COSTS` for the configured model (all Jev versions
use the Jev rate). Estimates exclude cache discounts and usage from failed
requests. Unpriced OpenAI models report token totals with null costs. Report-only
runs use the model saved in `experiment.json` and need no API key.

Regenerate reports without API calls:

```bash
uv run main.py --report-only
uv run main.py --method openai --report-only
```

To evaluate manually supplied passage runs, put six-column TREC `.gz`, `.txt`,
`.trec`, or `.run` files in `data/runs/` and use `--local-runs` (at least two
runs). Future judging methods can register in `METHODS` and return the same
judgment schema; evaluation and plots are shared.

For more on installing uv, see: [https://docs.astral.sh/uv/](https://docs.astral.sh/uv/)

## Build the TREC DL 2023 judged passage dataset

> [!Note]
> There's no need to build this dataset unless you want your own version.
>  
> The main script pulls the hosted version rather than rebuilding it.

Authenticate with `hf auth login` or set `HF_TOKEN` in `.env`, then run:

```bash
uv run --env-file .env scripts/make-trec-dl-2023-judged.py owner/repo
```

## Verifying the judged dataset

Run:

```bash
uv run scripts/summarize-trec-dl-2023-judged.py
```

This should output the following information:

```
Number of queries: 82
Number of unique passages: 21,873
Number of judgments: 22,327
Average query length (words): 6.84
Minimum query length (words): 2
Maximum query length (words): 15
Average judged passages per query: 272.28

Relevance grade | Judgments | Average judged passages per query
              0 |    13,866 | 169.10
              1 |     4,372 | 53.32
              2 |     2,259 | 27.55
              3 |     1,830 | 22.32
```

Which can be verified as correct using the original TREC DL 2023 paper: https://arxiv.org/pdf/2507.08890
