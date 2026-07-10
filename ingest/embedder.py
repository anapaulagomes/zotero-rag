import lancedb
from embed import embed_documents, embedding_dim
from lancedb.pydantic import LanceModel, Vector
from loguru import logger

TABLE_NAME = "documents"

# Below this row count, LanceDB's IVF_PQ index can't train a meaningful codebook and
# brute-force KNN is fast enough anyway, so we skip building an index.
MIN_ROWS_FOR_INDEX = 256


def _document_model() -> type[LanceModel]:
    """Build the documents schema with the configured embedding dimension.

    The vector width is fixed when the LanceDB table is created, so it can't be a module
    constant — it follows EMBED_DIM (owned by the embeddings package). Switching embedding
    model means a different dimension and a full re-ingest into a fresh table.
    """
    dim = embedding_dim()

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
        vector: Vector(dim)

    return Document


def ensure_table_dim(db_path: str) -> None:
    """Abort the run if the existing table was built with a different vector width.

    Without this check, switching EMBED_MODEL against an existing table fails one item at
    a time ("Provided schema does not match existing table schema") after each PDF has
    already been parsed, while already-ingested items keep their old-model vectors — a
    silently mixed, unusable index. Called by the orchestrator before any parsing starts.
    """
    db = lancedb.connect(db_path)
    if TABLE_NAME not in db.table_names():
        return
    table_dim = db.open_table(TABLE_NAME).schema.field("vector").type.list_size
    expected = embedding_dim()
    if table_dim != expected:
        raise SystemExit(
            f"Existing table at {db_path!r} holds {table_dim}-dim vectors, but the configured "
            f"embedding model needs EMBED_DIM={expected}. Switching embedding model requires a "
            "fresh table: point LANCEDB_PATH at a new directory (keeps the old index around) "
            "or delete the current one, then re-run the ingestion."
        )


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


def create_vector_index(db_path: str) -> None:
    """Build an ANN index on the documents table after ingestion.

    LanceDB falls back to brute-force KNN until an index exists — fine for a few thousand
    chunks, but latency grows linearly with the table. A failure here is non-fatal: queries
    still work (just unindexed), so we log and move on.
    """
    db = lancedb.connect(db_path)
    if TABLE_NAME not in db.table_names():
        return

    table = db.open_table(TABLE_NAME)
    rows = table.count_rows()
    if rows < MIN_ROWS_FOR_INDEX:
        logger.info(f"{rows} rows < {MIN_ROWS_FOR_INDEX}; skipping ANN index (brute force is fine)")
        return

    try:
        table.create_index(metric="l2", replace=True)
        logger.info(f"built ANN index over {rows} rows")
    except Exception as exc:
        logger.warning(f"ANN index build failed (queries still work via brute force): {exc}")


def embed_and_store(chunks: list[str], metadata: dict, db_path: str) -> int:
    """Embed chunks via the configured provider and insert them into the documents table.

    Idempotent: returns 0 without re-embedding if `metadata["item_id"]` already exists.
    """
    if not chunks:
        return 0

    document_model = _document_model()

    db = lancedb.connect(db_path)
    table = db.create_table(TABLE_NAME, schema=document_model, exist_ok=True)

    item_id = int(metadata["item_id"])
    if table.count_rows(filter=f"item_id = {item_id}") > 0:
        logger.debug(f"item_id {item_id} already embedded; skipping")
        return 0

    embeddings = embed_documents(chunks)

    records = [
        document_model(
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
