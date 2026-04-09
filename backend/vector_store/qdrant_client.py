from qdrant_client import QdrantClient as _QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)
import uuid

from config import settings
from embeddings.clip_encoder import get_embedding_dimension

# All modalities share CLIP size (text + image in one space).
_COLLECTION_NAMES = ("text", "image", "audio", "video")


class VectorStore:
    def __init__(self):
        self.client = _QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
        self._vector_size = get_embedding_dimension()
        self._ensure_collections()

    def _ensure_collections(self):
        existing = {c.name for c in self.client.get_collections().collections}
        for name in _COLLECTION_NAMES:
            if name in existing:
                info = self.client.get_collection(name)
                params = info.config.params.vectors
                cur: int | None = None
                if params is not None:
                    if hasattr(params, "size"):
                        cur = int(params.size)
                    elif isinstance(params, dict) and params:
                        p0 = next(iter(params.values()))
                        cur = int(getattr(p0, "size", 0)) or None
                if cur is not None and cur != self._vector_size:
                    raise RuntimeError(
                        f"Qdrant collection {name!r} has size {cur}, but "
                        f"{settings.hf_clip_model_id!r} uses {self._vector_size}. "
                        "Delete the collection or reset Qdrant, then restart."
                    )
                continue
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=self._vector_size,
                    distance=Distance.COSINE,
                ),
            )

    def upsert(self, collection: str, vectors: list[list[float]], payloads: list[dict]) -> list[str]:
        ids = [str(uuid.uuid4()) for _ in vectors]
        points = [
            PointStruct(id=id_, vector=vec, payload=payload)
            for id_, vec, payload in zip(ids, vectors, payloads)
        ]
        self.client.upsert(collection_name=collection, points=points)
        return ids

    def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int = 5,
        score_threshold: float = 0.3,
    ) -> list[dict]:
        results = self.client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        )
        return [
            {
                "id": str(r.id),
                "score": r.score,
                "payload": r.payload,
            }
            for r in results
        ]

    def search_many(
        self,
        collections: list[str],
        query_vectors: dict[str, list[float]],
        top_k: int = 5,
        score_threshold: float = 0.3,
    ) -> list[dict]:
        all_results = []
        for col in collections:
            if col not in query_vectors:
                continue
            hits = self.search(col, query_vectors[col], top_k=top_k, score_threshold=score_threshold)
            for h in hits:
                h["collection"] = col
            all_results.extend(hits)

        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:top_k]


_vs: VectorStore | None = None


class _LazyVectorStore:
    """Defer Qdrant + CLIP until first use so the API process can boot (e.g. `/docs` works)."""

    def __getattr__(self, name: str):
        global _vs
        if _vs is None:
            _vs = VectorStore()
        return getattr(_vs, name)


vector_store = _LazyVectorStore()
