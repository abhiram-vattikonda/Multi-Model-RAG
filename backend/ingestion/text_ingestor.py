from __future__ import annotations
import os
import re
from pathlib import Path
from pypdf import PdfReader
from embeddings.clip_encoder import embed_texts_clip
from vector_store.qdrant_client import vector_store

CHUNK_SIZE = 800     # characters
CHUNK_OVERLAP = 100


def _chunk_text(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, current = [], ""
    for sentence in sentences:
        if len(current) + len(sentence) < CHUNK_SIZE:
            current += " " + sentence
        else:
            if current:
                chunks.append(current.strip())
            current = sentence
    if current:
        chunks.append(current.strip())
    # Overlap: re-attach last sentence of prev chunk
    overlapped = []
    for i, chunk in enumerate(chunks):
        if i > 0:
            tail = chunks[i - 1][-CHUNK_OVERLAP:]
            chunk = tail + " " + chunk
        overlapped.append(chunk)
    return overlapped


class TextIngestor:
    def extract_chunks(self, file_path: str, filename: str) -> list[dict]:
        ext = Path(filename).suffix.lower()
        if ext == ".pdf":
            reader = PdfReader(file_path)
            pages = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                for chunk in _chunk_text(text):
                    if len(chunk.strip()) > 30:
                        pages.append({
                            "content": chunk,
                            "source_file": filename,
                            "media_type": "text",
                            "chunk_index": len(pages),
                            "page_number": i + 1,
                        })
            return pages
        else:
            with open(file_path, "r", errors="replace") as f:
                text = f.read()
            return [
                {
                    "content": chunk,
                    "source_file": filename,
                    "media_type": "text",
                    "chunk_index": i,
                    "page_number": None,
                }
                for i, chunk in enumerate(_chunk_text(text))
                if len(chunk.strip()) > 30
            ]

    def embed_chunks(self, chunks: list[dict]) -> list[list[float]]:
        texts = [c["content"] for c in chunks]
        return embed_texts_clip(texts)

    def store(self, chunks: list[dict], vectors: list[list[float]]) -> list[str]:
        return vector_store.upsert(
            collection="text",
            vectors=vectors,
            payloads=chunks,
        )