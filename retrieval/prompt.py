import re

from loguru import logger

SYSTEM_PROMPT = """\
You are a research assistant that answers questions based on excerpts from academic papers.

Each excerpt is prefixed with a source marker like [S1] and the paper's author and year.

RULES:
- Respond in the same language the user asked the question in.
- Base your answer ONLY on the excerpts provided. Do not invent information.
- Synthesize across excerpts: compare, contrast and connect findings from
  different papers when relevant.
- Cite inline using the source marker in square brackets exactly as given, e.g.
  [S1]. Prefer the author and year for readability, followed by the marker:
  (Silva et al., 2021) [S1].
- Cite ONLY the provided sources. Never cite a work mentioned inside an excerpt's
  text unless it is itself one of the provided sources.
- If the answer is not in the excerpts, say so explicitly (in the user's language)."""

USER_TEMPLATE = """\
EXCERPTS:
{chunks}

QUESTION: {query}"""

_BRACKET_GROUP = re.compile(r"\[([^\]]*)\]")
_MARKER = re.compile(r"S\d+")


def _markers_for(results: list[dict]) -> list[str]:
    """Assign one marker per result, shared across chunks of the same paper.

    Papers are keyed by item_id; chunks lacking one fall back to object identity so
    they never collapse together. Markers are numbered by first appearance: S1, S2, ...
    """
    assigned: dict[object, str] = {}
    markers = []
    for result in results:
        key = result.get("item_id")
        if key is None:
            key = id(result)
        if key not in assigned:
            assigned[key] = f"S{len(assigned) + 1}"
        markers.append(assigned[key])
    return markers


def build_user_prompt(query: str, results: list[dict]) -> str:
    """Compose the user message: retrieved chunks + question.

    Each excerpt is tagged with the source marker its paper will be cited by.
    System-level rules (role, citation format, language) live in SYSTEM_PROMPT.
    """
    markers = _markers_for(results)
    chunk_blocks = [
        f"[{marker}] {_format_citation(result)}\n{result['text']}"
        for marker, result in zip(markers, results, strict=True)
    ]
    return USER_TEMPLATE.format(chunks="\n\n".join(chunk_blocks), query=query)


def unique_sources(results: list[dict]) -> dict[str, dict]:
    """Map each source marker to the first chunk of its paper, in appearance order."""
    sources: dict[str, dict] = {}
    for marker, result in zip(_markers_for(results), results, strict=True):
        sources.setdefault(marker, result)
    return sources


def cited_markers(answer: str) -> list[str]:
    """Extract the source markers the model actually cited, in first-seen order.

    Only markers inside square brackets count, so prose like "the S1 protein" is
    ignored. Grouped citations such as [S1, S2] are split into individual markers.
    """
    seen: set[str] = set()
    ordered = []
    for group in _BRACKET_GROUP.findall(answer):
        for marker in _MARKER.findall(group):
            if marker not in seen:
                seen.add(marker)
                ordered.append(marker)
    return ordered


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
