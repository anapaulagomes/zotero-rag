import httpx
import lancedb
from lancedb.pydantic import LanceModel, Vector
from loguru import logger

EMBED_DIM = 768
TABLE_NAME = "documents"


class Document(LanceModel):
    chunk_id: str
    item_id: int
    item_type: str
    title: str
    author: str
    year: int | None
    journal: str | None
    doi: str | None
    url: str | None
    pdf_path: str | None
    text: str
    vector: Vector(EMBED_DIM)


def embed_and_store(
    chunks: list[str],
    metadata: dict,
    db_path: str,
    ollama_host: str,
    embed_model: str,
) -> int:
    """Embed chunks via Ollama and insert them into the documents table.

    Idempotent: returns 0 without re-embedding if `metadata["item_id"]` already exists.
    """
    if not chunks:
        return 0

    db = lancedb.connect(db_path)
    table = db.create_table(TABLE_NAME, schema=Document, exist_ok=True)

    item_id = int(metadata["item_id"])
    if table.count_rows(filter=f"item_id = {item_id}") > 0:
        logger.debug(f"item_id {item_id} already embedded; skipping")
        return 0

    embeddings = _embed_batch(chunks, ollama_host, embed_model)

    records = [
        Document(
            chunk_id=f"{item_id}_{index}",
            item_id=item_id,
            item_type=metadata["item_type"],
            title=metadata.get("title") or "",
            author=metadata.get("author") or "",
            year=metadata.get("year"),
            journal=metadata.get("journal"),
            doi=metadata.get("doi"),
            url=metadata.get("url"),
            pdf_path=metadata.get("pdf_path"),
            text=chunk,
            vector=vector,
        )
        for index, (chunk, vector) in enumerate(zip(chunks, embeddings, strict=True))
    ]

    table.add(records)
    return len(records)


def _embed_batch(chunks: list[str], ollama_host: str, embed_model: str) -> list[list[float]]:
    response = httpx.post(
        f"{ollama_host}/api/embed",
        json={"model": embed_model, "input": chunks},
        timeout=120.0,
    )
    response.raise_for_status()
    return response.json()["embeddings"]
