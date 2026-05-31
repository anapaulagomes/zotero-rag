import httpx
import lancedb
from lancedb.pydantic import LanceModel, Vector
from loguru import logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

EMBED_DIM = 768
TABLE_NAME = "documents"

# nomic-embed-text is trained with task prefixes; stored passages use "search_document:"
# and queries use "search_query:". Omitting them puts queries and documents in slightly
# different subspaces and measurably hurts retrieval.
DOCUMENT_PREFIX = "search_document: "


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


def existing_item_ids(db_path: str) -> set[int]:
    """Return the set of item_ids already present in the documents table.

    Used by the orchestrator to skip parsing for already-embedded items — far cheaper
    than letting the per-item check inside embed_and_store fire after `parse_document`
    has already done the expensive PDF work.
    """
    db = lancedb.connect(db_path)
    if TABLE_NAME not in db.table_names():
        return set()
    arrow_table = db.open_table(TABLE_NAME).to_lance().to_table(columns=["item_id"])
    return set(arrow_table.column("item_id").to_pylist())


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


# Retry only transient transport errors (connection refused, timeouts): a bad model
# name or malformed request raises HTTPStatusError and should fail fast instead.
@retry(
    retry=retry_if_exception_type(httpx.TransportError),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, max=30),
    reraise=True,
)
def _embed_batch(chunks: list[str], ollama_host: str, embed_model: str) -> list[list[float]]:
    inputs = [f"{DOCUMENT_PREFIX}{chunk}" for chunk in chunks]
    response = httpx.post(
        f"{ollama_host}/api/embed",
        json={"model": embed_model, "input": inputs},
        timeout=120.0,
    )
    response.raise_for_status()
    return response.json()["embeddings"]
