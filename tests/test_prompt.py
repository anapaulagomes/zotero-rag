from prompt import _format_citation, build_user_prompt


def test_single_author():
    assert _format_citation({"author": "Silva, João", "year": 2021}) == "(Silva, 2021)"


def test_two_authors_use_ampersand():
    citation = _format_citation({"author": "Silva, João; Costa, Maria", "year": 2020})
    assert citation == "(Silva & Costa, 2020)"


def test_three_or_more_authors_use_et_al():
    citation = _format_citation(
        {"author": "Silva, João; Costa, Maria; Souza, Ana", "year": 2019}
    )
    assert citation == "(Silva et al., 2019)"


def test_missing_year_is_omitted():
    assert _format_citation({"author": "Silva, João"}) == "(Silva)"


def test_missing_author_falls_back_to_unknown():
    assert _format_citation({"year": 2022}) == "(Unknown, 2022)"


def test_build_user_prompt_numbers_chunks_and_includes_query():
    results = [
        {"text": "First excerpt.", "author": "Silva, João", "year": 2021},
        {"text": "Second excerpt.", "author": "Costa, Maria", "year": 2020},
    ]
    prompt = build_user_prompt("What is X?", results)
    assert "[1] (Silva, 2021)" in prompt
    assert "[2] (Costa, 2020)" in prompt
    assert "What is X?" in prompt
    assert "First excerpt." in prompt
