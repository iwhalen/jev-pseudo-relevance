# /// script
# requires-python = ">=3.14"
# dependencies = ["datasets>=5.0.1", "huggingface-hub>=1.0"]
# ///

"""Download the judged TREC DL 2023 dataset and report summary statistics."""

import argparse
from collections import Counter
import json
from statistics import mean


def summarize(queries, passages, qrels):
    query_ids = set(queries["query_id"])
    if not query_ids or len(query_ids) != len(queries):
        raise ValueError("Queries must be nonempty and have unique IDs")
    passage_ids = set(passages["passage_id"])
    if len(passage_ids) != len(passages):
        raise ValueError("Passages must have unique IDs")
    lengths = [len(text.split()) for text in queries["text"]]
    grades = Counter()
    per_query = Counter()
    pairs = set()
    for row in qrels:
        qid, pid, grade = row["query_id"], row["passage_id"], row["relevance"]
        if qid not in query_ids or pid not in passage_ids:
            raise ValueError(f"Unknown judgment reference: {qid}, {pid}")
        if grade not in range(4):
            raise ValueError(f"Invalid relevance grade: {grade}")
        if (qid, pid) in pairs:
            raise ValueError(f"Duplicate judgment: {qid}, {pid}")
        pairs.add((qid, pid))
        grades[grade] += 1
        per_query[qid] += 1
    return {
        "number_of_queries": len(queries),
        "number_of_passages": len(passages),
        "number_of_judgments": len(qrels),
        "query_length_words": {
            "average": mean(lengths), "minimum": min(lengths), "maximum": max(lengths),
        },
        "average_judged_passages_per_query": mean(per_query[qid] for qid in query_ids),
        "relevance_grades": {
            str(grade): {
                "number_of_judgments": grades[grade],
                "average_judged_passages_per_query": grades[grade] / len(queries),
            }
            for grade in range(4)
        },
    }


def main():
    from datasets import load_dataset
    from huggingface_hub import HfApi

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="iwhalen/trec-dl-2023-judged")
    parser.add_argument("--revision", default="main", help="Hub branch, tag, or commit")
    parser.add_argument("--cache-dir", help="Hugging Face dataset download cache")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()
    revision = HfApi().dataset_info(args.repo_id, revision=args.revision).sha
    tables = {
        name: load_dataset(
            args.repo_id, name, split="test", revision=revision, cache_dir=args.cache_dir,
        )
        for name in ("queries", "passages", "qrels")
    }
    stats = summarize(**tables)
    if args.json:
        print(json.dumps({"repo_id": args.repo_id, "revision": revision, **stats}, indent=2))
        return
    print(f"Dataset: {args.repo_id} (revision {revision})")
    print(f"Number of queries: {stats['number_of_queries']:,}")
    print(f"Number of unique passages: {stats['number_of_passages']:,}")
    print(f"Number of judgments: {stats['number_of_judgments']:,}")
    lengths = stats["query_length_words"]
    print(f"Average query length (words): {lengths['average']:.2f}")
    print(f"Minimum query length (words): {lengths['minimum']}")
    print(f"Maximum query length (words): {lengths['maximum']}")
    print(f"Average judged passages per query: {stats['average_judged_passages_per_query']:.2f}")
    print("\nRelevance grade | Judgments | Average judged passages per query")
    for grade, values in stats["relevance_grades"].items():
        print(f"{grade:>15} | {values['number_of_judgments']:>9,} | "
              f"{values['average_judged_passages_per_query']:.2f}")
    print("\nWords are whitespace-separated. Grade averages include all queries, including zeros.")
    print("Documents here are judged passages, not unique parent documents.")


if __name__ == "__main__":
    main()
