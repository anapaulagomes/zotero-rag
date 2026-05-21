FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /workspace

COPY pyproject.toml uv.lock ./
COPY app/pyproject.toml app/

RUN uv sync --frozen --no-dev

COPY app/ app/

WORKDIR /workspace/app
CMD ["uv", "run", "chainlit", "run", "main.py", "--host", "0.0.0.0", "--port", "8000"]
