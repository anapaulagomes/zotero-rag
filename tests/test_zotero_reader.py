from pathlib import Path

from zotero_reader import _extract_year, _resolve_pdf_path


def test_extract_year_from_iso_date():
    assert _extract_year("2021-03-15") == 2021


def test_extract_year_from_freeform_string():
    assert _extract_year("March 2019") == 2019


def test_extract_year_handles_missing_value():
    assert _extract_year(None) is None
    assert _extract_year("") is None


def test_extract_year_returns_none_when_no_year_present():
    assert _extract_year("no date here") is None


def test_resolve_pdf_path_none_raw_path():
    assert _resolve_pdf_path(None, "ABCD1234", Path("/tmp")) is None


def test_resolve_managed_path_when_file_exists(tmp_path):
    storage_key = "ABCD1234"
    pdf_dir = tmp_path / storage_key
    pdf_dir.mkdir()
    pdf = pdf_dir / "paper.pdf"
    pdf.write_text("dummy")

    resolved = _resolve_pdf_path("storage:paper.pdf", storage_key, tmp_path)
    assert resolved == str(pdf)


def test_resolve_managed_path_missing_file_returns_none(tmp_path):
    assert _resolve_pdf_path("storage:missing.pdf", "ABCD1234", tmp_path) is None


def test_resolve_linked_path_when_file_exists(tmp_path):
    pdf = tmp_path / "linked.pdf"
    pdf.write_text("dummy")
    assert _resolve_pdf_path(str(pdf), "ignored", tmp_path) == str(pdf)


def test_resolve_linked_path_missing_returns_none(tmp_path):
    assert _resolve_pdf_path(str(tmp_path / "nope.pdf"), "ignored", tmp_path) is None
