from chunker import (
    MIN_CHUNK_CHARS,
    _split_recursive,
    _take_trailing_sentences,
    chunk_text,
)


def test_empty_text_yields_no_chunks():
    assert chunk_text("") == []


def test_short_text_below_minimum_is_dropped():
    assert chunk_text("too short") == []


def test_header_is_prefixed_to_each_chunk_from_its_section():
    body = "This is a sufficiently long sentence about surveillance systems. " * 3
    text = f"## Methods\n\n{body}"
    chunks = chunk_text(text, chunk_size=64, overlap=0)
    assert chunks
    assert all(chunk.startswith("## Methods") for chunk in chunks)


def test_chunks_respect_max_chars_budget():
    body = "word " * 2000
    chunk_size, overlap = 128, 0
    chunks = chunk_text(body, chunk_size=chunk_size, overlap=overlap)
    assert len(chunks) > 1
    # No chunk should exceed the char budget (chunk_size tokens * 4 chars/token).
    assert all(len(chunk) <= chunk_size * 4 for chunk in chunks)


def test_all_returned_chunks_meet_minimum_length():
    text = "## H\n\n" + ("Sentence number one is here. " * 50)
    chunks = chunk_text(text)
    assert chunks
    assert all(len(chunk) >= MIN_CHUNK_CHARS for chunk in chunks)


def test_split_recursive_hard_splits_a_single_oversized_token():
    giant = "x" * 1000
    pieces = _split_recursive(giant, max_chars=100)
    assert len(pieces) == 10
    assert all(len(piece) <= 100 for piece in pieces)


def test_take_trailing_sentences_snaps_to_sentence_boundary():
    text = "First sentence here. Second sentence follows on."
    # Window large enough to include the period, so it snaps to the sentence start.
    tail = _take_trailing_sentences(text, max_chars=30)
    assert tail.startswith("Second")


def test_take_trailing_sentences_falls_back_to_word_boundary():
    text = "First sentence here. Second sentence follows on."
    # Window excludes the period, so it can only snap to a word boundary.
    tail = _take_trailing_sentences(text, max_chars=25)
    assert not tail.startswith(" ")
    assert tail in text


def test_take_trailing_sentences_returns_all_when_short_enough():
    text = "Short."
    assert _take_trailing_sentences(text, max_chars=100) == text
