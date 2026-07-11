from prompt import _format_citation, build_user_prompt, cited_markers, unique_sources


def test_single_author():
    assert _format_citation({"author": "Silva, João", "year": 2021}) == "(Silva, 2021)"


def test_two_authors_use_ampersand():
    citation = _format_citation({"author": "Silva, João; Costa, Maria", "year": 2020})
    assert citation == "(Silva & Costa, 2020)"


def test_three_or_more_authors_use_et_al():
    citation = _format_citation({"author": "Silva, João; Costa, Maria; Souza, Ana", "year": 2019})
    assert citation == "(Silva et al., 2019)"


def test_missing_year_is_omitted():
    assert _format_citation({"author": "Silva, João"}) == "(Silva)"


def test_missing_author_falls_back_to_unknown():
    assert _format_citation({"year": 2022}) == "(Unknown, 2022)"


def test_build_user_prompt_marks_chunks_and_includes_query():
    results = [
        {"text": "First excerpt.", "author": "Silva, João", "year": 2021, "item_id": 1},
        {"text": "Second excerpt.", "author": "Costa, Maria", "year": 2020, "item_id": 2},
    ]
    prompt = build_user_prompt("What is X?", results)
    assert "[S1] (Silva, 2021)" in prompt
    assert "[S2] (Costa, 2020)" in prompt
    assert "What is X?" in prompt
    assert "First excerpt." in prompt


def test_chunks_from_the_same_paper_share_a_marker():
    results = [
        {"text": "First chunk.", "author": "Silva, João", "year": 2021, "item_id": 7},
        {"text": "Second chunk.", "author": "Silva, João", "year": 2021, "item_id": 7},
        {"text": "Other paper.", "author": "Costa, Maria", "year": 2020, "item_id": 9},
    ]
    prompt = build_user_prompt("What is X?", results)
    assert prompt.count("[S1]") == 2
    assert "[S2] (Costa, 2020)" in prompt


def test_unique_sources_maps_each_marker_to_the_first_chunk_of_its_paper():
    results = [
        {"text": "First chunk.", "author": "Silva, João", "year": 2021, "item_id": 7},
        {"text": "Second chunk.", "author": "Silva, João", "year": 2021, "item_id": 7},
        {"text": "Other paper.", "author": "Costa, Maria", "year": 2020, "item_id": 9},
    ]
    sources = unique_sources(results)
    assert list(sources) == ["S1", "S2"]
    assert sources["S1"]["text"] == "First chunk."
    assert sources["S2"]["author"] == "Costa, Maria"


def test_cited_markers_extracts_markers_in_first_seen_order():
    answer = "Fever surveillance works (Silva et al., 2021) [S2] and is limited [S1]."
    assert cited_markers(answer) == ["S2", "S1"]


def test_cited_markers_handles_grouped_citations_and_deduplicates():
    answer = "Both agree [S2, S1]. As noted before [S1], sensitivity is low."
    assert cited_markers(answer) == ["S2", "S1"]


def test_cited_markers_ignores_markers_without_brackets():
    answer = "The S1 protein and figure S3 are unrelated to citations."
    assert cited_markers(answer) == []
