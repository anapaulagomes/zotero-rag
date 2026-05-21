FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /workspace

COPY pyproject.toml uv.lock ./
COPY ingest/pyproject.toml ingest/

RUN uv sync --frozen --no-dev

COPY ingest/ ingest/

WORKDIR /workspace/ingest
CMD ["uv", "run", "python", "main.py"]
