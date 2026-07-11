import chainlit as cl
import lancedb
import polars as pl
from config import get_settings
from llm import stream_chat
from loguru import logger
from prompt import SYSTEM_PROMPT, build_user_prompt, cited_markers, unique_sources
from search import search


@cl.on_chat_start
async def on_chat_start() -> None:
    await cl.Message(content=_index_summary()).send()


def _index_summary() -> str:
    empty_message = "Index is empty. Run `ingest/main.py` first."
    try:
        db = lancedb.connect(get_settings().lancedb_path)
        table = db.open_table("documents")
    except Exception:
        return empty_message

    total_chunks = table.count_rows()
    if total_chunks == 0:
        return empty_message

    df = pl.from_arrow(table.to_arrow()).select(["item_id", "title", "year"])
    # item_id ascends with Zotero insertion order, so a descending sort surfaces the
    # most recently added items (not necessarily the newest by publication year).
    recently_added = df.unique(subset=["item_id"]).sort("item_id", descending=True)

    sample_lines = []
    for row in recently_added.head(5).iter_rows(named=True):
        title = (row["title"] or "(untitled)")[:90]
        year = row["year"] if row["year"] is not None else "?"
        sample_lines.append(f"- *{title}* ({year})")

    return (
        f"**{len(recently_added)} papers** in the index ({total_chunks} chunks).\n\n"
        "Recently added:\n" + "\n".join(sample_lines)
    )


@cl.on_message
async def on_message(message: cl.Message) -> None:
    query = message.content
    settings = get_settings()

    try:
        results = search(
            query,
            settings.lancedb_path,
            top_k=settings.top_k,
            min_score=settings.score_threshold,
        )
    except Exception as exc:
        logger.exception("search failed")
        await cl.Message(content=f"Search failed: {exc}").send()
        return

    if not results:
        await cl.Message(content="No relevant results found in the library.").send()
        return

    user_prompt = build_user_prompt(query, results)
    response_message = cl.Message(content="")

    answer = ""
    try:
        async for token in stream_chat(SYSTEM_PROMPT, user_prompt):
            answer += token
            await response_message.stream_token(token)
    except Exception as exc:
        logger.exception("llm stream failed")
        await cl.Message(content=f"LLM call failed: {exc}").send()
        return

    cited = _cited_sources(results, answer)

    # Attaching a Text element named after each marker turns every [S7] in the answer
    # into a link that opens that reference in the side panel (Chainlit matches element
    # names against the message text). Longer markers win, so S1 never grabs S10/S13.
    response_message.elements = [
        cl.Text(name=marker, content=_reference_entry(marker, result), display="side")
        for marker, result in cited
    ]
    await response_message.send()

    references = _format_references(cited)
    if references:
        await cl.Message(content=references).send()


def _cited_sources(results: list[dict], answer: str) -> list[tuple[str, dict]]:
    """The (marker, paper) pairs the answer actually cited, in marker order.

    Markers come from `unique_sources` (one per paper); `cited_markers` reads back
    which ones the model used, so uncited retrieved papers stay out.
    """
    cited = set(cited_markers(answer))
    return [
        (marker, result) for marker, result in unique_sources(results).items() if marker in cited
    ]


def _reference_entry(marker: str, result: dict) -> str:
    author = result.get("author") or "Unknown author"
    year = result.get("year") or "n.d."
    title = result.get("title") or "(untitled)"
    journal = result.get("journal") or ""
    doi = result.get("doi") or ""
    pdf_path = result.get("pdf_path")
    url = result.get("url") or ""

    entry = f"**[{marker}] {author}** ({year}) — *{title}*"
    if journal:
        entry += f", {journal}"
    if doi:
        entry += f"\n   doi: [{doi}](https://doi.org/{doi})"
    if url:
        entry += f"\n   {url}"
    if pdf_path:
        entry += f"\n   `{pdf_path}`"
    return entry


def _format_references(cited: list[tuple[str, dict]]) -> str:
    if not cited:
        return ""

    lines = ["**References**", ""]
    for marker, result in cited:
        lines.append(_reference_entry(marker, result))
        lines.append("")
    return "\n".join(lines)
