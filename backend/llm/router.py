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
    model = settings.llm_model
    for existing_prefix in _PROVIDER_PREFIX.values():
        if existing_prefix and model.startswith(existing_prefix):
            if settings.llm_provider == "ollama":
                litellm.api_base = settings.ollama_base_url
            return model
    prefix = _PROVIDER_PREFIX.get(settings.llm_provider, "")
    if settings.llm_provider == "ollama":
        litellm.api_base = settings.ollama_base_url
    return f"{prefix}{model}"


DEFAULT_SYSTEM = (
    "You are a helpful assistant. Answer the user's question using ONLY the "
    "provided context chunks. Cite the source file when referencing specific content. "
    "If the context is insufficient, say so honestly. "
    "Format answers so they are easy to scan in plain text: start with a short direct answer, "
    "then use short section headings and bullet points when listing items, steps, categories, or comparisons. "
    "Do not compress list answers into one dense paragraph. "
    "Group related items together, keep each bullet to one idea, and leave a blank line between major sections."
)


def _supports_vision() -> bool:
    model = _get_model().lower()
    return "llama-4-scout" in model


def _chunk_image_data_url(chunk: dict) -> str | None:
    payload = chunk.get("payload", chunk)
    image_base64 = payload.get("image_base64")
    if not image_base64:
        return None
    mime_type = payload.get("image_mime_type") or "image/jpeg"
    return f"data:{mime_type};base64,{image_base64}"


def _build_user_content(query: str, context_chunks: list[dict]) -> str | list[dict]:
    context = _build_context(context_chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {query}"
    if not _supports_vision():
        return prompt

    content: list[dict] = [{"type": "text", "text": prompt}]
    image_count = 0
    for chunk in context_chunks:
        image_url = _chunk_image_data_url(chunk)
        if not image_url:
            continue
        content.append({"type": "image_url", "image_url": {"url": image_url}})
        image_count += 1
        if image_count >= 5:
            break
    return content


async def generate_stream(
    query: str,
    context_chunks: list[dict],
    system_prompt: str | None = None,
) -> AsyncIterator[str]:
    messages = [
        {"role": "system", "content": system_prompt or DEFAULT_SYSTEM},
        {"role": "user", "content": _build_user_content(query, context_chunks)},
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
    messages = [
        {"role": "system", "content": system_prompt or DEFAULT_SYSTEM},
        {"role": "user", "content": _build_user_content(query, context_chunks)},
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
