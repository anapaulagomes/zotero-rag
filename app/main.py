import chainlit as cl
import lancedb
import polars as pl
from config import get_settings
from llm import stream_chat
from loguru import logger
from prompt import SYSTEM_PROMPT, build_user_prompt
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
    unique_papers = df.unique(subset=["item_id"]).sort("item_id", descending=True)

    sample_lines = []
    for row in unique_papers.head(5).iter_rows(named=True):
        title = (row["title"] or "(untitled)")[:90]
        year = row["year"] if row["year"] is not None else "?"
        sample_lines.append(f"- *{title}* ({year})")

    return (
        f"**{len(unique_papers)} papers** in the index ({total_chunks} chunks).\n\n"
        "Most recent:\n" + "\n".join(sample_lines)
    )


@cl.on_message
async def on_message(message: cl.Message) -> None:
    query = message.content
    settings = get_settings()

    try:
        results = search(
            query,
            settings.lancedb_path,
            settings.ollama_host,
            settings.embed_model,
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

    try:
        async for token in stream_chat(SYSTEM_PROMPT, user_prompt):
            await response_message.stream_token(token)
    except Exception as exc:
        logger.exception("llm stream failed")
        await cl.Message(content=f"LLM call failed: {exc}").send()
        return

    await response_message.send()
    await cl.Message(content=_format_references(results)).send()


def _format_references(results: list[dict]) -> str:
    seen: set[int] = set()
    lines = ["**References**", ""]
    for result in results:
        item_id = result.get("item_id")
        if item_id in seen:
            continue
        seen.add(item_id)

        author = result.get("author") or "Unknown author"
        year = result.get("year") or "n.d."
        title = result.get("title") or "(untitled)"
        journal = result.get("journal") or ""
        doi = result.get("doi") or ""
        pdf_path = result.get("pdf_path")
        url = result.get("url") or ""

        entry = f"**{author}** ({year}) — *{title}*"
        if journal:
            entry += f", {journal}"
        if doi:
            entry += f"\n   doi: [{doi}](https://doi.org/{doi})"
        if url:
            entry += f"\n   {url}"
        if pdf_path:
            entry += f"\n   `{pdf_path}`"
        lines.append(entry)
        lines.append("")

    return "\n".join(lines)
