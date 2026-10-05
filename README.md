# Jev pseudo-relevance

To run the experiments, first get your `TYPESAFE_API_KEY` set in a `.env` file.

Then run:

```bash
uv run --env-file .env main.py
```

Outputs will be written to `data/`.

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
