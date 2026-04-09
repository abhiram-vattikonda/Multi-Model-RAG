from __future__ import annotations
import os
import litellm
from config import settings
from typing import AsyncIterator

# Configure provider API keys
os.environ["GROQ_API_KEY"] = settings.groq_api_key
os.environ["OPENAI_API_KEY"] = settings.openai_api_key
os.environ["ANTHROPIC_API_KEY"] = settings.anthropic_api_key

# Provider → litellm model prefix mapping
_PROVIDER_PREFIX = {
    "groq": "groq/",
    "openai": "",
    "anthropic": "anthropic/",
    "ollama": "ollama/",
}


def _get_model() -> str:
    prefix = _PROVIDER_PREFIX.get(settings.llm_provider, "")
    model = settings.llm_model
    if settings.llm_provider == "ollama":
        litellm.api_base = settings.ollama_base_url
    return f"{prefix}{model}"


DEFAULT_SYSTEM = (
    "You are a helpful assistant. Answer the user's question using ONLY the "
    "provided context chunks. Cite the source file when referencing specific content. "
    "If the context is insufficient, say so honestly."
)


async def generate_stream(
    query: str,
    context_chunks: list[dict],
    system_prompt: str | None = None,
) -> AsyncIterator[str]:
    context = _build_context(context_chunks)
    messages = [
        {"role": "system", "content": system_prompt or DEFAULT_SYSTEM},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ]
    response = await litellm.acompletion(
        model=_get_model(),
        messages=messages,
        stream=True,
        max_tokens=2048,
    )
    async for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


async def generate(
    query: str,
    context_chunks: list[dict],
    system_prompt: str | None = None,
) -> str:
    context = _build_context(context_chunks)
    messages = [
        {"role": "system", "content": system_prompt or DEFAULT_SYSTEM},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ]
    response = await litellm.acompletion(
        model=_get_model(),
        messages=messages,
        stream=False,
        max_tokens=2048,
    )
    return response.choices[0].message.content


def _build_context(chunks: list[dict]) -> str:
    lines = []
    for i, chunk in enumerate(chunks, 1):
        payload = chunk.get("payload", chunk)
        source = payload.get("source_file", "unknown")
        content = payload.get("content", "")
        ts = ""
        if payload.get("timestamp_start") is not None:
            ts = f" [{payload['timestamp_start']:.1f}s - {payload['timestamp_end']:.1f}s]"
        lines.append(f"[{i}] ({source}{ts}): {content}")
    return "\n".join(lines)


def get_provider_info() -> dict:
    return {
        "provider": settings.llm_provider,
        "model": settings.llm_model,
        "full_model_string": _get_model(),
    }