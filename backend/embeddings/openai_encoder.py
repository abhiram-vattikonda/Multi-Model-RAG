from openai import OpenAI
from config import settings

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


def embed_texts(texts: list[str], model: str | None = None) -> list[list[float]]:
    model = model or "text-embedding-3-small"
    client = _get_client()
    # OpenAI allows batches of up to 2048 inputs
    all_embeddings = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(input=batch, model=model)
        all_embeddings.extend([d.embedding for d in resp.data])
    return all_embeddings


def embed_text(text: str, model: str | None = None) -> list[float]:
    return embed_texts([text], model)[0]