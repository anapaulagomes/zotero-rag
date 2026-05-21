FROM python:3.12-slim

# Native libraries docling and rapidocr need on slim images:
#   libxcb1, fontconfig, libfreetype6 → pypdfium2 page rendering
#   libgl1, libglib2.0-0              → rapidocr (image preprocessing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxcb1 \
    fontconfig \
    libfreetype6 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /workspace

COPY pyproject.toml uv.lock ./
COPY ingest/pyproject.toml ingest/

RUN uv sync --frozen --no-dev

COPY ingest/ ingest/

WORKDIR /workspace/ingest
CMD ["uv", "run", "python", "main.py"]
