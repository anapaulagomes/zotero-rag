from loguru import logger

SYSTEM_PROMPT = """\
You are a research assistant that answers questions based on excerpts from academic papers.

RULES:
- Respond in the same language the user asked the question in.
- Base your answer ONLY on the excerpts provided. Do not invent information.
- Synthesize across excerpts: compare, contrast and connect findings from
  different papers when relevant.
- Cite authors and year inline, in the format: (Silva et al., 2021).
- If the answer is not in the excerpts, say so explicitly (in the user's language)."""

USER_TEMPLATE = """\
EXCERPTS:
{chunks}

QUESTION: {query}"""


def build_user_prompt(query: str, results: list[dict]) -> str:
    """Compose the user message: retrieved chunks + question.

    System-level rules (role, citation format, language) live in SYSTEM_PROMPT.
    """
    chunk_blocks = [
        f"[trecho {index}] {_format_citation(result)}\n{result['text']}"
        for index, result in enumerate(results, start=1)
    ]
    return USER_TEMPLATE.format(chunks="\n\n".join(chunk_blocks), query=query)


def _format_citation(result: dict) -> str:
    author_raw = result.get("author") or ""
    year = result.get("year")

    authors = [a.strip() for a in author_raw.split(";") if a.strip()]
    surnames = [a.split(",")[0].strip() for a in authors]
    surnames = [s for s in surnames if s]

    # APA: one author -> surname; two -> "A & B"; three or more -> "A et al."
    if not surnames:
        citation = "Unknown"
    elif len(surnames) == 1:
        citation = surnames[0]
    elif len(surnames) == 2:
        citation = f"{surnames[0]} & {surnames[1]}"
    else:
        citation = f"{surnames[0]} et al."

    return f"({citation}, {year})" if year else f"({citation})"


if __name__ == "__main__":
    mock_results = [
        {
            "text": "Syndromic surveillance systems detect outbreaks using chief complaints.",
            "author": "Chapman, Wendy; Dowling, John",
            "year": 2007,
        },
        {
            "text": "Chief complaints alone have low sensitivity for febrile syndromes.",
            "author": "Smith, John",
            "year": 2020,
        },
    ]
    rendered = build_user_prompt("O que são sistemas de vigilância sindrômica?", mock_results)
    logger.info("system prompt:\n{}", SYSTEM_PROMPT)
    logger.info("user prompt:\n{}", rendered)
