from types import SimpleNamespace

import embed
import pytest


def _fake_response(vectors: list[list[float]]) -> SimpleNamespace:
    """Mimic LiteLLM's EmbeddingResponse: a `.data` list of {"embedding": vector}."""
    return SimpleNamespace(data=[{"embedding": vector} for vector in vectors])


def test_nomic_gets_task_prefixes(monkeypatch):
    monkeypatch.setenv("EMBED_MODEL", "nomic-embed-text")
    assert embed._task_prefixes() == ("search_document: ", "search_query: ")


def test_nomic_prefix_keyed_on_bare_name_ignoring_ollama_tag(monkeypatch):
    monkeypatch.setenv("EMBED_MODEL", "nomic-embed-text:latest")
    assert embed._task_prefixes() == ("search_document: ", "search_query: ")


def test_unlisted_model_gets_no_prefix(monkeypatch):
    monkeypatch.setenv("EMBED_MODEL", "bge-m3")
    assert embed._task_prefixes() == ("", "")


def test_resolve_ollama_passes_host_as_api_base(monkeypatch):
    monkeypatch.setenv("EMBED_PROVIDER", "ollama")
    monkeypatch.setenv("EMBED_MODEL", "nomic-embed-text")
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
    model, kwargs = embed._resolve()
    assert model == "ollama/nomic-embed-text"
    assert kwargs == {"api_base": "http://localhost:11434"}


def test_resolve_openai_uses_embed_base_url(monkeypatch):
    monkeypatch.setenv("EMBED_PROVIDER", "openai")
    monkeypatch.setenv("EMBED_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("EMBED_BASE_URL", "http://localhost:8080/v1")
    model, kwargs = embed._resolve()
    assert model == "openai/text-embedding-3-small"
    assert kwargs == {"api_base": "http://localhost:8080/v1"}


def test_resolve_rejects_unknown_provider(monkeypatch):
    monkeypatch.setenv("EMBED_PROVIDER", "cohere")
    monkeypatch.setenv("EMBED_MODEL", "embed-v4")
    with pytest.raises(ValueError, match="Unsupported EMBED_PROVIDER"):
        embed._resolve()


def test_embed_documents_applies_document_prefix(monkeypatch):
    monkeypatch.setenv("EMBED_PROVIDER", "ollama")
    monkeypatch.setenv("EMBED_MODEL", "nomic-embed-text")
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
    monkeypatch.setenv("EMBED_DIM", "3")

    captured = {}

    def fake_embedding(model, input, **kwargs):
        captured["input"] = input
        return _fake_response([[0.0, 0.0, 0.0] for _ in input])

    monkeypatch.setattr(embed, "embedding", fake_embedding)

    embed.embed_documents(["hello", "world"])
    assert captured["input"] == ["search_document: hello", "search_document: world"]


def test_embed_query_applies_query_prefix_and_unwraps(monkeypatch):
    monkeypatch.setenv("EMBED_PROVIDER", "ollama")
    monkeypatch.setenv("EMBED_MODEL", "nomic-embed-text")
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
    monkeypatch.setenv("EMBED_DIM", "3")

    captured = {}

    def fake_embedding(model, input, **kwargs):
        captured["input"] = input
        return _fake_response([[1.0, 2.0, 3.0]])

    monkeypatch.setattr(embed, "embedding", fake_embedding)

    vector = embed.embed_query("what is syndromic surveillance")
    assert captured["input"] == ["search_query: what is syndromic surveillance"]
    assert vector == [1.0, 2.0, 3.0]


def test_dimension_mismatch_raises(monkeypatch):
    monkeypatch.setenv("EMBED_PROVIDER", "ollama")
    monkeypatch.setenv("EMBED_MODEL", "nomic-embed-text")
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
    monkeypatch.setenv("EMBED_DIM", "1024")

    def fake_embedding(model, input, **kwargs):
        return _fake_response([[0.0] * 768])

    monkeypatch.setattr(embed, "embedding", fake_embedding)

    with pytest.raises(ValueError, match="returned 768-dim vectors but EMBED_DIM=1024"):
        embed.embed_query("mismatch")
