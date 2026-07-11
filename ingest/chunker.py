import re
import sys
from pathlib import Path

from loguru import logger

CHARS_PER_TOKEN = 4
MIN_CHUNK_CHARS = 100
RECURSIVE_SEPARATORS = ["\n\n", "\n", ". ", " "]
HEADER_PATTERN = re.compile(r"^#{1,4}\s+\S")
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[str]:
    """Split markdown text into header-aware, recursively-split chunks with sentence-aware overlap.

    Each chunk that comes from a section carries the section header as a prefix to preserve
    retrieval context. chunk_size and overlap are in tokens (approximated as chars / 4).
    Chunks shorter than 100 chars are dropped.
    """
    max_chars = chunk_size * CHARS_PER_TOKEN
    overlap_chars = overlap * CHARS_PER_TOKEN

    sections = _split_into_sections(text)
    if not sections:
        return []

    final_chunks: list[str] = []
    for header, body in sections:
        # Reserve budget for the header so the final chunk roughly fits max_chars.
        body_budget = max_chars - len(header) - 2 if header else max_chars
        body_pieces = _split_recursive(body, body_budget)
        body_pieces = _apply_overlap(body_pieces, overlap_chars)
        for piece in body_pieces:
            final_chunks.append(f"{header}\n\n{piece}" if header else piece)

    return [chunk for chunk in final_chunks if len(chunk) >= MIN_CHUNK_CHARS]


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """Walk markdown line by line, grouping content under its nearest preceding header."""
    sections: list[tuple[str, str]] = []
    current_header = ""
    current_body: list[str] = []

    def flush() -> None:
        body = "\n".join(current_body).strip()
        if body:
            sections.append((current_header, body))

    for line in text.split("\n"):
        if HEADER_PATTERN.match(line):
            flush()
            current_header = line.strip()
            current_body = []
        else:
            current_body.append(line)
    flush()

    return sections


def _split_recursive(text: str, max_chars: int) -> list[str]:
    """Recursively split text and greedily pack pieces into chunks of <= max_chars."""
    if len(text) <= max_chars:
        return [text] if text.strip() else []

    for separator in RECURSIVE_SEPARATORS:
        if separator not in text:
            continue

        parts = text.split(separator)
        chunks: list[str] = []
        buffer = ""

        for part in parts:
            candidate = f"{buffer}{separator}{part}" if buffer else part
            if len(candidate) <= max_chars:
                buffer = candidate
                continue

            if buffer:
                chunks.append(buffer)
                buffer = ""

            if len(part) <= max_chars:
                buffer = part
            else:
                chunks.extend(_split_recursive(part, max_chars))

        if buffer:
            chunks.append(buffer)
        return chunks

    # Last resort: hard char split (only reached when a single "word" exceeds max_chars).
    return [
        text[start : start + max_chars]
        for start in range(0, len(text), max_chars)
        if text[start : start + max_chars].strip()
    ]


def _apply_overlap(chunks: list[str], overlap_chars: int) -> list[str]:
    """Prepend the trailing sentences of chunk N-1 to chunk N, capped by overlap_chars."""
    if overlap_chars <= 0 or len(chunks) <= 1:
        return chunks

    result = [chunks[0]]
    for index in range(1, len(chunks)):
        tail = _take_trailing_sentences(chunks[index - 1], overlap_chars)
        result.append(f"{tail}\n\n{chunks[index]}" if tail else chunks[index])
    return result


def _take_trailing_sentences(text: str, max_chars: int) -> str:
    """Take the tail of text up to max_chars, snapped forward to a sentence or word boundary."""
    if len(text) <= max_chars:
        return text

    tail = text[-max_chars:]
    sentence_match = SENTENCE_BOUNDARY.search(tail)
    if sentence_match:
        return tail[sentence_match.end() :]
    space_index = tail.find(" ")
    return tail[space_index + 1 :] if space_index != -1 else tail


if __name__ == "__main__":
    if len(sys.argv) != 2:
        logger.error("usage: python chunker.py <text-file>")
        sys.exit(1)

    content = Path(sys.argv[1]).read_text(encoding="utf-8")
    chunks = chunk_text(content)
    logger.info(f"produced {len(chunks)} chunks from {len(content)} chars")
    for index, chunk in enumerate(chunks[:3], start=1):
        logger.info(f"chunk {index} ({len(chunk)} chars):\n{chunk[:300]}...")

    if len(chunks) >= 2:
        logger.info("--- overlap check ---")
        logger.info(f"chunk 1 tail: ...{chunks[0][-150:]!r}")
        logger.info(f"chunk 2 head: {chunks[1][:150]!r}...")
