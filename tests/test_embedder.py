import lancedb
import pyarrow as pa
import pytest
from embedder import TABLE_NAME, ensure_table_dim


def _create_table(db_path: str, dim: int) -> None:
    schema = pa.schema(
        [
            pa.field("item_id", pa.int64()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
        ]
    )
    lancedb.connect(db_path).create_table(TABLE_NAME, schema=schema)


def test_no_table_yet_passes(tmp_path, monkeypatch):
    monkeypatch.setenv("EMBED_DIM", "1024")
    ensure_table_dim(str(tmp_path))


def test_matching_dim_passes(tmp_path, monkeypatch):
    monkeypatch.setenv("EMBED_DIM", "768")
    _create_table(str(tmp_path), 768)
    ensure_table_dim(str(tmp_path))


def test_mismatched_dim_aborts(tmp_path, monkeypatch):
    monkeypatch.setenv("EMBED_DIM", "1024")
    _create_table(str(tmp_path), 768)
    with pytest.raises(SystemExit, match="768-dim vectors.*EMBED_DIM=1024"):
        ensure_table_dim(str(tmp_path))
