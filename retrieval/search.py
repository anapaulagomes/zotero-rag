import os
import sys

import httpx
import lancedb
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

TABLE_NAME = "documents"

# Must match the passage prefix used at ingestion time (see ingest/embedder.py).
QUERY_PREFIX = "search_query: "


def search(
    query: str,
    db_path: str,
    ollama_host: str,
    embed_model: str,
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
    query_vector = _embed_query(query, ollama_host, embed_model)

    db = lancedb.connect(db_path)
    table = db.open_table(TABLE_NAME)

    results = table.search(query_vector).limit(top_k).to_list()

    for record in results:
        record["score"] = 1.0 / (1.0 + record.pop("_distance"))
        record.pop("vector", None)

    return [record for record in results if record["score"] >= min_score]


def _embed_query(query: str, ollama_host: str, embed_model: str) -> list[float]:
    response = httpx.post(
        f"{ollama_host}/api/embed",
        json={"model": embed_model, "input": f"{QUERY_PREFIX}{query}"},
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()["embeddings"][0]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.error("usage: python search.py '<query>'")
        sys.exit(1)

    user_query = " ".join(sys.argv[1:])
    hits = search(
        query=user_query,
        db_path=os.environ["LANCEDB_PATH"],
        ollama_host=os.environ["OLLAMA_HOST"],
        embed_model=os.environ["EMBED_MODEL"],
    )

    logger.info(f"top {len(hits)} results for: {user_query!r}")
    for index, hit in enumerate(hits, start=1):
        title = (hit.get("title") or "")[:80]
        logger.info(f"[{index}] score={hit['score']:.3f} | {title} ({hit.get('year', '?')})")
        logger.info(f"    {hit['text'][:200]}...")
