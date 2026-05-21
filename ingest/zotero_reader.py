import os
import re
import shutil
import sqlite3
import tempfile
from pathlib import Path

import polars as pl
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

DEFAULT_ZOTERO_DB = Path(os.environ.get("ZOTERO_DB") or Path.home() / "Zotero" / "zotero.sqlite")
DEFAULT_STORAGE = Path(os.environ.get("ZOTERO_STORAGE") or Path.home() / "Zotero" / "storage")

YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")

ITEMS_QUERY = """
SELECT
    i.itemID,
    i.key,
    it.typeName                                     AS item_type,
    MAX(CASE WHEN f.fieldName = 'title'
             THEN idv.value END)                    AS title,
    MAX(CASE WHEN f.fieldName IN ('date', 'year')
             THEN idv.value END)                    AS date,
    MAX(CASE WHEN f.fieldName = 'publicationTitle'
             THEN idv.value END)                    AS journal,
    MAX(CASE WHEN f.fieldName = 'DOI'
             THEN idv.value END)                    AS doi,
    MAX(CASE WHEN f.fieldName = 'url'
             THEN idv.value END)                    AS url
FROM items i
JOIN itemTypes it      ON i.itemTypeID  = it.itemTypeID
LEFT JOIN itemData id  ON i.itemID      = id.itemID
LEFT JOIN fields f     ON id.fieldID    = f.fieldID
LEFT JOIN itemDataValues idv ON id.valueID = idv.valueID
WHERE it.typeName NOT IN ('attachment', 'note')
  AND i.itemID NOT IN (SELECT itemID FROM deletedItems)
GROUP BY i.itemID
"""

AUTHORS_QUERY = """
SELECT
    ic.itemID,
    c.lastName  || ', ' || c.firstName AS author
FROM itemCreators ic
JOIN creators c     ON ic.creatorID     = c.creatorID
JOIN creatorTypes ct ON ic.creatorTypeID = ct.creatorTypeID
WHERE ct.creatorType = 'author'
ORDER BY ic.itemID, ic.orderIndex
"""

ATTACHMENTS_QUERY = """
SELECT
    ia.parentItemID AS itemID,
    i.key           AS storage_key,
    ia.path         AS raw_path,
    ia.contentType
FROM itemAttachments ia
JOIN items i ON ia.itemID = i.itemID
WHERE ia.contentType = 'application/pdf'
  AND ia.parentItemID IS NOT NULL
"""


def _resolve_pdf_path(raw_path: str, storage_key: str, storage_dir: Path) -> str | None:
    if raw_path is None:
        return None

    # Zotero stores either 'storage:filename.pdf' (managed) or an absolute path (linked file).
    if raw_path.startswith("storage:"):
        filename = raw_path.removeprefix("storage:")
        managed_path = storage_dir / storage_key / filename
        return str(managed_path) if managed_path.exists() else None

    linked_path = Path(raw_path)
    return str(linked_path) if linked_path.exists() else None


def _extract_year(date_str: str | None) -> int | None:
    if not date_str:
        return None
    for part in date_str.split("-"):
        part = part.strip()
        if len(part) == 4 and part.isdigit():
            return int(part)
    match = YEAR_PATTERN.search(date_str)
    return int(match.group()) if match else None


def read_library(
    db_path: Path | str = DEFAULT_ZOTERO_DB,
    storage_dir: Path | str = DEFAULT_STORAGE,
    only_with_pdf: bool = True,
) -> pl.DataFrame:
    """Read the Zotero SQLite database and return one row per item.

    Columns:
        item_id    int       internal Zotero ID
        item_type  str       journalArticle, book, etc.
        title      str
        author     str       "Silva, João; Costa, Maria"
        year       int|null
        journal    str|null
        doi        str|null
        url        str|null
        pdf_path   str|null  absolute path to the PDF file

    Args:
        db_path: path to zotero.sqlite
        storage_dir: Zotero storage/ directory
        only_with_pdf: drop items without a resolved PDF path
    """
    db_path = Path(db_path)
    storage_dir = Path(storage_dir)

    if not db_path.exists():
        raise FileNotFoundError(f"Zotero database not found: {db_path}")

    # Copy to a temp file so we don't conflict with a running Zotero instance.
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    shutil.copy2(db_path, tmp_path)

    try:
        connection = sqlite3.connect(f"file:{tmp_path}?mode=ro", uri=True)
        items_rows = connection.execute(ITEMS_QUERY).fetchall()
        authors_rows = connection.execute(AUTHORS_QUERY).fetchall()
        attachments_rows = connection.execute(ATTACHMENTS_QUERY).fetchall()
        connection.close()
    finally:
        tmp_path.unlink(missing_ok=True)

    items_df = pl.DataFrame(
        items_rows,
        schema=["item_id", "key", "item_type", "title", "date", "journal", "doi", "url"],
        orient="row",
    )

    authors_df = (
        pl.DataFrame(authors_rows, schema=["item_id", "author"], orient="row")
        .group_by("item_id")
        .agg(pl.col("author").str.join("; "))
    )

    resolved_attachments = []
    for item_id, storage_key, raw_path, _content_type in attachments_rows:
        pdf_path = _resolve_pdf_path(raw_path, storage_key, storage_dir)
        resolved_attachments.append({"item_id": item_id, "pdf_path": pdf_path})

    attachments_df = (
        pl.DataFrame(resolved_attachments)
        .filter(pl.col("pdf_path").is_not_null())
        .group_by("item_id")
        .first()
    )

    library_df = (
        items_df.join(authors_df, on="item_id", how="left")
        .join(attachments_df, on="item_id", how="left")
        .drop("key")
        .with_columns(
            pl.col("date").map_elements(_extract_year, return_dtype=pl.Int32).alias("year")
        )
        .drop("date")
        .select(
            [
                "item_id",
                "item_type",
                "title",
                "author",
                "year",
                "journal",
                "doi",
                "url",
                "pdf_path",
            ]
        )
        .sort("year", descending=True, nulls_last=True)
    )

    if only_with_pdf:
        library_df = library_df.filter(pl.col("pdf_path").is_not_null())

    return library_df


if __name__ == "__main__":
    library_df = read_library()
    logger.info(f"{len(library_df)} items with a resolved PDF")
    logger.info("\n{}", library_df.select(["title", "author", "year", "journal"]).head(10))

    logger.info("Distribution by item type:")
    logger.info("\n{}", library_df.group_by("item_type").len().sort("len", descending=True))

    full_library_df = read_library(only_with_pdf=False)
    items_without_pdf = full_library_df.filter(pl.col("pdf_path").is_null())
    logger.info(f"{len(items_without_pdf)} items without an associated PDF")
