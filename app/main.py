import os

import chainlit as cl
import lancedb
import polars as pl
from dotenv import load_dotenv
from llm import stream_chat
from loguru import logger
from prompt import SYSTEM_PROMPT, build_user_prompt
from search import search

load_dotenv()

LANCEDB_PATH = os.environ["LANCEDB_PATH"]
OLLAMA_HOST = os.environ["OLLAMA_HOST"]
EMBED_MODEL = os.environ["EMBED_MODEL"]
TOP_K = int(os.environ.get("TOP_K", "15"))
SCORE_THRESHOLD = float(os.environ.get("SCORE_THRESHOLD", "0.0"))


@cl.on_chat_start
async def on_chat_start() -> None:
    await cl.Message(content=_index_summary()).send()


def _index_summary() -> str:
    empty_message = "Index is empty. Run `ingest/main.py` first."
    try:
        db = lancedb.connect(LANCEDB_PATH)
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

    try:
        results = search(
            query,
            LANCEDB_PATH,
            OLLAMA_HOST,
            EMBED_MODEL,
            top_k=TOP_K,
            min_score=SCORE_THRESHOLD,
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
