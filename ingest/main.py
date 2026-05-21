import os
import time

from chunker import chunk_text
from dotenv import load_dotenv
from embedder import embed_and_store
from loguru import logger
from parser import parse_document
from tqdm import tqdm
from zotero_reader import read_library

load_dotenv()


def _build_chunks(metadata: dict) -> list[str]:
    """Pick the best available content for an item and apply the title prefix.

    Cascade: pdf body -> abstract -> title-as-stub. The title is also prepended to every
    chunk (except in the title-only stub case, where it would duplicate). Returns [] when
    the item has no usable content.
    """
    title = metadata.get("title") or ""
    pdf_path = metadata.get("pdf_path")
    abstract = metadata.get("abstract")

    chunks: list[str] = []
    if pdf_path:
        try:
            text = parse_document(pdf_path)
            chunks = chunk_text(text)
        except Exception as exc:
            logger.warning(f"parse_document failed for {pdf_path}: {exc}")

    if not chunks and abstract:
        chunks = [abstract]

    if not chunks and title:
        return [title]

    if not chunks:
        return []

    if title:
        return [f"{title}\n\n{chunk}" for chunk in chunks]
    return chunks


def main() -> None:
    db_path = os.environ["LANCEDB_PATH"]
    ollama_host = os.environ["OLLAMA_HOST"]
    embed_model = os.environ["EMBED_MODEL"]

    library = read_library()
    total_items = len(library)

    start = time.monotonic()
    total_inserted = 0
    failed = 0

    for row in tqdm(library.iter_rows(named=True), total=total_items, desc="ingesting"):
        try:
            chunks = _build_chunks(row)
            if not chunks:
                continue
            inserted = embed_and_store(
                chunks=chunks,
                metadata=row,
                db_path=db_path,
                ollama_host=ollama_host,
                embed_model=embed_model,
            )
            total_inserted += inserted
        except Exception as exc:
            failed += 1
            logger.warning(f"item_id={row.get('item_id')}: {exc}")

    elapsed = time.monotonic() - start
    logger.info(
        f"done — {total_items} items processed, "
        f"{total_inserted} new chunks, {failed} failures, "
        f"{elapsed:.0f}s elapsed"
    )


if __name__ == "__main__":
    main()
