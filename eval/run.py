"""Retrieval eval: measure recall@k and MRR against the golden set.

Runs each question in eval/questions.yaml through retrieval.search() using the
CURRENTLY-CONFIGURED embedding model (EMBED_MODEL / EMBED_DIM) and index
(LANCEDB_PATH), then scores whether the relevant item_ids surface in the top-k.
No LLM is involved — this isolates retrieval quality, which is what decides one
embedding model vs another.

Run it once per index to compare (swap EMBED_MODEL+EMBED_DIM+LANCEDB_PATH together):

    # nomic (768-dim)
    EMBED_MODEL=nomic-embed-text EMBED_DIM=768 LANCEDB_PATH=./data/lancedb \
        uv run python eval/run.py

    # bge-m3 (1024-dim)
    EMBED_MODEL=bge-m3 EMBED_DIM=1024 LANCEDB_PATH=./data/lancedb-bge-m3 \
        uv run python eval/run.py

Metrics (a question counts as a hit if ANY of its relevant item_ids is retrieved):
  recall@k  fraction of questions with >=1 relevant item in the top-k
  MRR       mean reciprocal rank of the FIRST relevant item (0 if not found)

Reported overall and split by language, so you can see the cross-lingual gap
directly. Exit code is always 0; this is a report, not a gate.
"""

import os
import re
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from loguru import logger

# retrieval/ is a workspace package exposing search(); import it the same way the
# app does. Adding the repo's retrieval dir to the path keeps this script runnable
# with a plain `uv run` without extra install steps.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "retrieval"))
from search import search  # noqa: E402

load_dotenv()

QUESTIONS_PATH = Path(__file__).resolve().parent / "questions.yaml"
TOP_K = int(os.environ.get("EVAL_TOP_K", "15"))


_DOI_RE = re.compile(r"^10\.\d{4,9}/", re.IGNORECASE)


def _norm_doi(value: str | None) -> str:
    """Normalize a DOI for comparison: drop any resolver prefix, casefold.

    Zotero stores DOIs inconsistently — bare (`10.1/x`), URL-prefixed
    (`https://doi.org/10.1/x`), or mixed-case — so compare on a canonical form.
    """
    doi = (value or "").strip().casefold()
    doi = re.sub(r"^(https?://)?(dx\.)?doi\.org/", "", doi)
    return doi.removeprefix("doi:").strip()


def _norm_title(value: str | None) -> str:
    """Normalize a title for comparison: casefold, collapse whitespace, drop a
    trailing period."""
    return re.sub(r"\s+", " ", (value or "").strip().casefold()).rstrip(".")


def _split_relevant(relevant: list[str]) -> tuple[set[str], set[str]]:
    """Partition golden-set entries into normalized DOIs and normalized titles.

    An entry is treated as a DOI if it looks like one (`10.xxxx/...`, optionally
    resolver-prefixed), otherwise as an exact title.
    """
    dois, titles = set(), set()
    for entry in relevant:
        if _DOI_RE.match(_norm_doi(entry)):
            dois.add(_norm_doi(entry))
        else:
            titles.add(_norm_title(entry))
    return dois, titles


def _first_relevant_rank(hits: list[dict], dois: set[str], titles: set[str]) -> int | None:
    """1-based rank of the first retrieved chunk matching a relevant DOI or title."""
    for rank, hit in enumerate(hits, start=1):
        if _norm_doi(hit.get("doi")) in dois or _norm_title(hit.get("title")) in titles:
            return rank
    return None


def evaluate(questions: list[dict], db_path: str) -> list[dict]:
    rows = []
    for q in questions:
        dois, titles = _split_relevant(q["relevant"])
        hits = search(query=q["question"], db_path=db_path, top_k=TOP_K)
        rank = _first_relevant_rank(hits, dois, titles)
        rows.append(
            {
                "id": q["id"],
                "lang": q["lang"],
                "hit": rank is not None,
                "rank": rank,
                "rr": 1.0 / rank if rank else 0.0,
            }
        )
        marker = f"rank {rank}" if rank else "MISS"
        logger.info(f"[{q['id']}] ({q['lang']}) {marker:>8}  {q['question'][:60]}")
    return rows


def _summarize(rows: list[dict], label: str) -> None:
    if not rows:
        return
    n = len(rows)
    recall = sum(r["hit"] for r in rows) / n
    mrr = sum(r["rr"] for r in rows) / n
    logger.info(f"{label:<12} n={n:<3} recall@{TOP_K}={recall:.3f}  MRR={mrr:.3f}")


def main() -> None:
    db_path = os.environ["LANCEDB_PATH"]
    model = os.environ.get("EMBED_MODEL", "?")
    dim = os.environ.get("EMBED_DIM", "?")

    data = yaml.safe_load(QUESTIONS_PATH.read_text())
    questions = data["questions"]

    logger.info(f"eval: model={model} dim={dim} index={db_path} top_k={TOP_K} n={len(questions)}")
    rows = evaluate(questions, db_path)

    logger.info("--- results ---")
    _summarize([r for r in rows if r["lang"] == "pt"], "pt-br")
    _summarize([r for r in rows if r["lang"] == "en"], "en")
    _summarize(rows, "overall")


if __name__ == "__main__":
    main()
