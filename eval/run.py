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


def _first_relevant_rank(hits: list[dict], relevant: set[str]) -> int | None:
    """1-based rank of the first retrieved chunk whose item belongs to `relevant`."""
    for rank, hit in enumerate(hits, start=1):
        if str(hit.get("item_id")) in relevant:
            return rank
    return None


def evaluate(questions: list[dict], db_path: str) -> list[dict]:
    rows = []
    for q in questions:
        relevant = {str(item_id) for item_id in q["relevant"]}
        hits = search(query=q["question"], db_path=db_path, top_k=TOP_K)
        rank = _first_relevant_rank(hits, relevant)
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
