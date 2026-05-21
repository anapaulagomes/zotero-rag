import os
import sys
from functools import cache
from pathlib import Path

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from loguru import logger

SUPPORTED_EXTENSIONS = {".pdf", ".html", ".htm"}


@cache
def _get_converter() -> DocumentConverter:
    # Force CPU: docling's layout model (rt_detr_v2) builds float64 position embeddings,
    # and MPS doesn't support float64. Auto-detect would pick MPS on Apple Silicon and crash.
    # num_threads scales with the host (8 on M1, 10 on M4); falls back to docling's default of 4.
    pdf_options = PdfPipelineOptions()
    pdf_options.accelerator_options = AcceleratorOptions(
        device=AcceleratorDevice.CPU,
        num_threads=os.cpu_count() or 4,
    )
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)},
    )


def parse_document(path: str) -> str:
    """Return extracted text as markdown. Raises ValueError if the format is unsupported."""
    file_path = Path(path)

    if not file_path.exists():
        logger.warning(f"file not found, skipping: {path}")
        return ""

    extension = file_path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"unsupported format '{extension}' for {path}")

    result = _get_converter().convert(file_path)
    return result.document.export_to_markdown()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        logger.error("usage: python parser.py <file-or-directory>")
        sys.exit(1)

    target = Path(sys.argv[1])

    if target.is_dir():
        files = sorted(f for f in target.rglob("*") if f.suffix.lower() in SUPPORTED_EXTENSIONS)
        logger.info(f"found {len(files)} supported files under {target}")
        for index, file_path in enumerate(files, start=1):
            extracted = parse_document(str(file_path))
            logger.info(f"[{index}/{len(files)}] {file_path.name}: {len(extracted)} chars")
    else:
        extracted_text = parse_document(str(target))
        logger.info(f"extracted {len(extracted_text)} characters")
        logger.info("preview:\n{}", extracted_text[:1000])
