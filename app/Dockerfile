FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /workspace

# Member pyprojects are needed to resolve the workspace graph; the retrieval source is
# needed because the app depends on it as a built workspace package.
COPY pyproject.toml uv.lock ./
COPY app/pyproject.toml app/
COPY ingest/pyproject.toml ingest/
COPY retrieval/ retrieval/

RUN uv sync --frozen --no-dev --package zotero-rag-app

COPY app/ app/

WORKDIR /workspace/app
CMD ["uv", "run", "chainlit", "run", "main.py", "--host", "0.0.0.0", "--port", "8000"]
