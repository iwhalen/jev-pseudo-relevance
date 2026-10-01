"""Main entry point."""

import gzip
import hashlib
import json
import re
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory

from datasets import Dataset, DatasetDict, Features, Value, load_from_disk
from datasets.arrow_writer import ArrowWriter
from tqdm.auto import tqdm

DATA_PATH = Path(__file__).resolve().parent / "data"
QUERIES_PATH = DATA_PATH / "2023_queries.tsv"
QRELS_PATH = DATA_PATH / "qrels.dl23-doc-msmarco-v2.1.txt"
CORPUS_PATH = DATA_PATH / "corpus"
CORPUS_ARCHIVE = DATA_PATH / "msmarco_v2.1.doc.tar"


def load_queries(path: Path) -> Dataset:
    ids, texts = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        qid, text = line.split("\t", 1)
        ids.append(qid)
        texts.append(text)
    return Dataset.from_dict({"id": ids, "text": texts})


def load_qrels(path: Path) -> Dataset:
    query_ids, corpus_ids, scores = [], [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        qid, _, docid, score = line.split()
        query_ids.append(qid)
        corpus_ids.append(docid)
        scores.append(int(score))
    return Dataset.from_dict(
        {"query-id": query_ids, "corpus-id": corpus_ids, "score": scores}
    )


DOC_ID = re.compile(r"msmarco_v2\.1_doc_(\d{2})_(\d+)")
FEATURES = Features({column: Value("string") for column in ("id", "title", "text")})


def validate_inputs(queries: Dataset, qrels: Dataset) -> set[str]:
    needed = set(qrels["corpus-id"])
    invalid = sorted(docid for docid in needed if not DOC_ID.fullmatch(docid))
    if invalid:
        raise ValueError(
            f"Qrels are incompatible with MS MARCO V2.1: {invalid[:3]}. "
            "Use data/qrels.dl23-doc-msmarco-v2.1.txt; V2 offsets cannot be renamed."
        )
    unknown = set(qrels["query-id"]) - set(queries["id"])
    if unknown:
        raise ValueError(f"Qrels reference unknown queries: {sorted(unknown)[:5]}")
    return needed


def iter_documents(archive_path: Path, needed: set[str]):
    """Stream only required shards, retaining at most one JSON record at a time."""
    if not needed:
        return
    pending = {}
    for docid in needed:
        shard = DOC_ID.fullmatch(docid).group(1)
        pending.setdefault(shard, set()).add(docid)
    sampled = False
    with (
        tarfile.open(archive_path, "r|", stream=True) as archive,
        tqdm(total=len(needed), desc="Judged documents", unit="doc") as progress,
    ):
        for member in archive:
            if not member.isfile() or not member.name.endswith(".json.gz"):
                continue
            shard = Path(member.name).name.removesuffix(".json.gz").rsplit("_", 1)[-1]
            targets = pending.get(shard)
            # Inspect one record even when the first shard is irrelevant.
            if not targets and sampled:
                continue
            with (
                archive.extractfile(member) as compressed,
                gzip.GzipFile(fileobj=compressed) as records,
            ):
                for line in records:
                    doc = json.loads(line)
                    if not sampled:
                        if not DOC_ID.fullmatch(doc.get("docid", "")):
                            raise ValueError(
                                "Archive is not an MS MARCO V2.1 document corpus."
                            )
                        sampled = True
                    if not targets:
                        break
                    if doc["docid"] not in targets:
                        continue
                    targets.remove(doc["docid"])
                    progress.update(1)
                    yield {
                        "id": doc["docid"],
                        "title": doc["title"],
                        "text": "\n".join(
                            part for part in (doc["headings"], doc["body"]) if part
                        ),
                    }
                    if not targets:
                        del pending[shard]
                        break
            if not pending:
                return
    missing = sorted(docid for targets in pending.values() for docid in targets)
    raise ValueError(
        f"Archive is missing {len(missing)} judged documents in shards "
        f"{sorted(pending)}; examples: {missing[:5]}"
    )


def load_corpus(needed: set[str]) -> Dataset:
    stat = CORPUS_ARCHIVE.stat()
    manifest = {
        "schema_version": 1,
        "qrels_sha256": hashlib.sha256(QRELS_PATH.read_bytes()).hexdigest(),
        "archive": str(CORPUS_ARCHIVE.resolve()),
        "archive_size": stat.st_size,
        "archive_mtime_ns": stat.st_mtime_ns,
    }
    key = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    # Versioned directories preserve existing caches and publish by atomic rename.
    CORPUS_PATH.mkdir(parents=True, exist_ok=True)
    destination = CORPUS_PATH / key
    if destination.exists():
        stored = json.loads((destination / "manifest.json").read_text())
        corpus = load_from_disk(destination / "dataset")
        if stored == manifest and set(corpus["id"]) == needed:
            return corpus
        raise ValueError(f"Invalid corpus cache: {destination}")
    with TemporaryDirectory(prefix=".building-", dir=CORPUS_PATH) as temporary:
        stage = Path(temporary)
        if needed:
            arrow_path = stage / "documents.arrow"
            with ArrowWriter(
                path=str(arrow_path), features=FEATURES, writer_batch_size=100
            ) as writer:
                for document in iter_documents(CORPUS_ARCHIVE, needed):
                    writer.write(document)
                writer.finalize()
            corpus = Dataset.from_file(str(arrow_path))
        else:
            corpus = Dataset.from_dict(
                {column: [] for column in FEATURES}, features=FEATURES
            )
        if set(corpus["id"]) != needed:
            raise ValueError("Built corpus does not cover the qrels.")
        publication = stage / "complete"
        publication.mkdir()
        corpus.save_to_disk(publication / "dataset")
        (publication / "manifest.json").write_text(json.dumps(manifest, indent=2))
        publication.rename(destination)
    return load_from_disk(destination / "dataset")


def load_local_dataset() -> DatasetDict:
    """Load aligned local V2.1 inputs and a disk-backed judged-document corpus."""
    queries = load_queries(QUERIES_PATH)
    qrels = load_qrels(QRELS_PATH)
    needed = validate_inputs(queries, qrels)
    judged_query_ids = set(qrels["query-id"])
    queries = queries.filter(lambda query: query["id"] in judged_query_ids)
    corpus = load_corpus(needed)
    return DatasetDict({"qrels": qrels, "queries": queries, "corpus": corpus})


def main():
    data = load_local_dataset()
    print(data)
    data.save_to_disk(DATA_PATH / "msmarco-v2.1-trec-dl2023-judged")


if __name__ == "__main__":
    main()
