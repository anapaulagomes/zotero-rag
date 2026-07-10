import os
import sys

import lancedb
from dotenv import load_dotenv
from embed import embed_query
from loguru import logger

load_dotenv()

TABLE_NAME = "documents"


def search(
    query: str,
    db_path: str,
    top_k: int = 15,
    min_score: float = 0.0,
) -> list[dict]:
    """Embed query and return top-k chunks from LanceDB.

    Each result includes text, title, author, year, journal, doi, url, pdf_path,
    plus a normalized `score` (higher = closer). The raw `_distance` is replaced.

    Chunks scoring below `min_score` are dropped, so low-relevance context never
    reaches the LLM (and never shows up in the references). The default keeps every
    hit; raise it once you've measured the score distribution on your own library.
    """
    query_vector = embed_query(query)

    db = lancedb.connect(db_path)
    table = db.open_table(TABLE_NAME)

    results = table.search(query_vector).limit(top_k).to_list()

    for record in results:
        record["score"] = 1.0 / (1.0 + record.pop("_distance"))
        record.pop("vector", None)

    return [record for record in results if record["score"] >= min_score]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.error("usage: python search.py '<query>'")
        sys.exit(1)

    user_query = " ".join(sys.argv[1:])
    hits = search(query=user_query, db_path=os.environ["LANCEDB_PATH"])

    logger.info(f"top {len(hits)} results for: {user_query!r}")
    for index, hit in enumerate(hits, start=1):
        title = (hit.get("title") or "")[:80]
        logger.info(f"[{index}] score={hit['score']:.3f} | {title} ({hit.get('year', '?')})")
        logger.info(f"    {hit['text'][:200]}...")
