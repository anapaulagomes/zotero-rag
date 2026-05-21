import json
import os
import sys
from pathlib import Path

import chainlit as cl
import httpx
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

_RETRIEVAL_DIR = os.environ.get("RETRIEVAL_PATH") or str(
    Path(__file__).resolve().parent.parent / "retrieval"
)
sys.path.insert(0, _RETRIEVAL_DIR)

from prompt import build_prompt  # noqa: E402
from search import search  # noqa: E402

LANCEDB_PATH = os.environ["LANCEDB_PATH"]
OLLAMA_HOST = os.environ["OLLAMA_HOST"]
EMBED_MODEL = os.environ["EMBED_MODEL"]
LLM_MODEL = os.environ["LLM_MODEL"]
TOP_K = 5


@cl.on_message
async def on_message(message: cl.Message) -> None:
    query = message.content

    try:
        results = search(query, LANCEDB_PATH, OLLAMA_HOST, EMBED_MODEL, top_k=TOP_K)
    except Exception as exc:
        logger.exception("search failed")
        await cl.Message(content=f"Erro na busca: {exc}").send()
        return

    if not results:
        await cl.Message(content="Não encontrei nada relevante na biblioteca.").send()
        return

    prompt = build_prompt(query, results)
    response_message = cl.Message(content="")

    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            async with client.stream(
                "POST",
                f"{OLLAMA_HOST}/api/chat",
                json={
                    "model": LLM_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": True,
                },
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    payload = json.loads(line)
                    piece = payload.get("message", {}).get("content", "")
                    if piece:
                        await response_message.stream_token(piece)
    except Exception as exc:
        logger.exception("ollama chat failed")
        await cl.Message(content=f"Erro no Ollama: {exc}").send()
        return

    await response_message.send()
    await cl.Message(content=_format_references(results)).send()


def _format_references(results: list[dict]) -> str:
    seen: set[int] = set()
    lines = ["**📚 Referências**", ""]
    for result in results:
        item_id = result.get("item_id")
        if item_id in seen:
            continue
        seen.add(item_id)

        author = result.get("author") or "Autor desconhecido"
        year = result.get("year") or "s.d."
        title = result.get("title") or "(sem título)"
        journal = result.get("journal") or ""
        doi = result.get("doi") or ""
        pdf_path = result.get("pdf_path")
        url = result.get("url") or ""

        entry = f"📄 **{author}** ({year}) — *{title}*"
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
