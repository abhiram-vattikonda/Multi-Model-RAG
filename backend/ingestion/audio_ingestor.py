from __future__ import annotations
import whisper
from embeddings.clip_encoder import embed_texts_clip
from vector_store.qdrant_client import vector_store

_whisper_model = None

def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model("base")
    return _whisper_model


class AudioIngestor:
    def extract_chunks(self, file_path: str, filename: str) -> list[dict]:
        model = _get_whisper()
        result = model.transcribe(file_path, word_timestamps=False)
        chunks = []
        for i, segment in enumerate(result["segments"]):
            text = segment["text"].strip()
            if len(text) > 10:
                chunks.append({
                    "content": text,
                    "source_file": filename,
                    "media_type": "audio",
                    "chunk_index": i,
                    "timestamp_start": segment["start"],
                    "timestamp_end": segment["end"],
                })
        return chunks

    def embed_chunks(self, chunks: list[dict]) -> list[list[float]]:
        texts = [c["content"] for c in chunks]
        return embed_texts_clip(texts)

    def store(self, chunks: list[dict], vectors: list[list[float]]) -> list[str]:
        return vector_store.upsert(
            collection="audio",
            vectors=vectors,
            payloads=chunks,
        )