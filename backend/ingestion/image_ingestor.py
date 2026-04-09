from __future__ import annotations
from PIL import Image
from embeddings.clip_encoder import embed_image_pil, get_embedding_dimension
from vector_store.qdrant_client import vector_store


class ImageIngestor:
    def extract_chunks(self, file_path: str, filename: str) -> list[dict]:
        image = Image.open(file_path).convert("RGB")
        w, h = image.size
        return [{
            "content": f"Image: {filename} ({w}x{h}px)",
            "source_file": filename,
            "media_type": "image",
            "chunk_index": 0,
            "image_path": file_path,
            "_image": image,   # temp, stripped before storing
        }]

    def embed_chunks(self, chunks: list[dict]) -> list[list[float]]:
        dim = get_embedding_dimension()
        embeddings = []
        for chunk in chunks:
            img = chunk.get("_image")
            if img:
                embeddings.append(embed_image_pil(img))
            else:
                embeddings.append([0.0] * dim)
        return embeddings

    def store(self, chunks: list[dict], vectors: list[list[float]]) -> list[str]:
        payloads = [{k: v for k, v in c.items() if k != "_image"} for c in chunks]
        return vector_store.upsert(
            collection="image",
            vectors=vectors,
            payloads=payloads,
        )