import time
from collections import Counter

from chunker import chunk_text
from config import get_settings
from embedder import create_vector_index, embed_and_store, ensure_table_dim, existing_item_ids
from loguru import logger
from parser import parse_document
from tqdm import tqdm
from zotero_reader import read_library


def _build_chunks(metadata: dict) -> tuple[list[str], str]:
    """Pick the best available content for an item and apply the title prefix.

    Cascade: pdf body -> abstract -> title-as-stub. The title is also prepended to every
    chunk (except in the title-only stub case, where it would duplicate). Returns
    (chunks, source) where source is 'pdf', 'abstract', 'title', or 'empty'.
    """
    title = metadata.get("title") or ""
    pdf_path = metadata.get("pdf_path")
    abstract = metadata.get("abstract")

    chunks: list[str] = []
    source = "empty"

    if pdf_path:
        try:
            text = parse_document(pdf_path)
            chunks = chunk_text(text)
            if chunks:
                source = "pdf"
        except Exception as exc:
            logger.warning(f"parse_document failed for {pdf_path}: {exc}")

    if not chunks and abstract:
        chunks = [abstract]
        source = "abstract"

    if not chunks and title:
        return [title], "title"

    if not chunks:
        return [], "empty"

    if title:
        chunks = [f"{title}\n\n{chunk}" for chunk in chunks]
    return chunks, source


def _route_loguru_through_tqdm() -> None:
    """Without this, loguru lines collide with tqdm's bar redraw and corrupt the display."""
    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""))


def main() -> None:
    _route_loguru_through_tqdm()

    settings = get_settings()
    db_path = settings.lancedb_path

    ensure_table_dim(db_path)

    library = read_library()
    total_items = len(library)

    already_done = existing_item_ids(db_path)
    if already_done:
        logger.info(f"{len(already_done)} items already in the index — will skip them")

    start = time.monotonic()
    total_inserted = 0
    failed = 0
    sources: Counter[str] = Counter()

    for row in tqdm(library.iter_rows(named=True), total=total_items, desc="ingesting"):
        item_id = int(row["item_id"])
        if item_id in already_done:
            sources["already_done"] += 1
            continue
        try:
            chunks, source = _build_chunks(row)
            sources[source] += 1
            if not chunks:
                continue
            inserted = embed_and_store(chunks=chunks, metadata=row, db_path=db_path)
            total_inserted += inserted
        except Exception as exc:
            failed += 1
            logger.warning(f"item_id={item_id}: {exc}")

    create_vector_index(db_path)

    elapsed = time.monotonic() - start
    logger.info(
        f"done in {elapsed:.0f}s | {total_items} items "
        f"(pdf={sources['pdf']}, abstract={sources['abstract']}, "
        f"title={sources['title']}, empty={sources['empty']}, "
        f"already_done={sources['already_done']}) | "
        f"{total_inserted} new chunks | {failed} failures"
    )


if __name__ == "__main__":
    main()
