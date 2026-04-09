import json
import hashlib
import redis
from config import settings

_redis = redis.from_url(settings.redis_url, decode_responses=True)

QUERY_CACHE_TTL = 3600   # 1 hour


def _make_key(query: str, top_k: int, modalities: list[str]) -> str:
    raw = f"{query}|{top_k}|{','.join(sorted(modalities))}"
    return "rag:query:" + hashlib.sha256(raw.encode()).hexdigest()


def get_cached_result(query: str, top_k: int, modalities: list[str]) -> dict | None:
    key = _make_key(query, top_k, modalities)
    val = _redis.get(key)
    if val:
        return json.loads(val)
    return None


def cache_result(query: str, top_k: int, modalities: list[str], result: dict) -> None:
    key = _make_key(query, top_k, modalities)
    _redis.setex(key, QUERY_CACHE_TTL, json.dumps(result))


def invalidate_all() -> int:
    keys = _redis.keys("rag:query:*")
    if keys:
        return _redis.delete(*keys)
    return 0