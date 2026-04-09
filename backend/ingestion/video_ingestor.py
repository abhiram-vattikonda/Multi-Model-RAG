from __future__ import annotations
import os
import tempfile
import whisper
from PIL import Image
from moviepy.editor import VideoFileClip
from embeddings.clip_encoder import embed_image_pil, embed_texts_clip
from vector_store.qdrant_client import vector_store

_whisper_model = None

def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model("base")
    return _whisper_model


KEYFRAME_INTERVAL = 5   # extract a frame every N seconds


class VideoIngestor:
    def extract_audio(self, video_path: str) -> str:
        tmp = tempfile.mktemp(suffix=".mp3")
        clip = VideoFileClip(video_path)
        clip.audio.write_audiofile(tmp, logger=None)
        clip.close()
        return tmp

    def transcribe(self, audio_path: str, filename: str) -> list[dict]:
        model = _get_whisper()
        result = model.transcribe(audio_path, word_timestamps=False)
        chunks = []
        for i, seg in enumerate(result["segments"]):
            text = seg["text"].strip()
            if len(text) > 10:
                chunks.append({
                    "content": text,
                    "source_file": filename,
                    "media_type": "video",
                    "chunk_index": i,
                    "chunk_type": "transcript",
                    "timestamp_start": seg["start"],
                    "timestamp_end": seg["end"],
                })
        if os.path.exists(audio_path):
            os.remove(audio_path)
        return chunks

    def extract_keyframes(self, video_path: str, filename: str) -> list[dict]:
        clip = VideoFileClip(video_path)
        duration = int(clip.duration)
        chunks = []
        for t in range(0, duration, KEYFRAME_INTERVAL):
            frame = clip.get_frame(t)
            pil_img = Image.fromarray(frame)
            chunks.append({
                "content": f"Video frame at {t}s from {filename}",
                "source_file": filename,
                "media_type": "video",
                "chunk_index": t,
                "chunk_type": "keyframe",
                "timestamp_start": float(t),
                "timestamp_end": float(min(t + KEYFRAME_INTERVAL, duration)),
                "_image": pil_img,
            })
        clip.close()
        return chunks

    def embed_chunks(self, chunks: list[dict]) -> list[list[float]]:
        embeddings = []
        for chunk in chunks:
            if chunk.get("_image"):
                vec = embed_image_pil(chunk["_image"])
            else:
                vec = embed_texts_clip([chunk["content"]])[0]
            embeddings.append(vec)
        return embeddings

    def store(self, chunks: list[dict], vectors: list[list[float]]) -> list[str]:
        payloads = [{k: v for k, v in c.items() if k != "_image"} for c in chunks]
        return vector_store.upsert(
            collection="video",
            vectors=vectors,
            payloads=payloads,
        )