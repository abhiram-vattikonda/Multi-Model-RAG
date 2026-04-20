from __future__ import annotations

import os
import re
import uuid
from pathlib import Path
from typing import Literal

from celery.result import AsyncResult
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from config import settings
from embeddings.clip_encoder import embed_text_clip
from llm.router import generate, generate_stream, get_provider_info
from models.schemas import (
    ChunkMetadata,
    GenerateRequest,
    GenerateResponse,
    IngestResponse,
    MediaType,
    RetrievalRequest,
    RetrievalResponse,
    RetrievedChunk,
    TaskStatusResponse,
)
from tasks.celery_app import celery_app
from tasks.ingest_tasks import ingest_audio, ingest_image, ingest_text, ingest_video
from vector_store.qdrant_client import vector_store

_TEXT_EXT = {".txt", ".md", ".pdf", ".json", ".csv", ".html", ".htm", ".xml"}
_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
_AUDIO_EXT = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}
_VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in",
    "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "what",
    "when", "where", "which", "who", "why", "with",
}


def _detect_media_type(filename: str) -> MediaType:
    ext = Path(filename).suffix.lower()
    if ext in _TEXT_EXT:
        return MediaType.text
    if ext in _IMAGE_EXT:
        return MediaType.image
    if ext in _AUDIO_EXT:
        return MediaType.audio
    if ext in _VIDEO_EXT:
        return MediaType.video
    raise HTTPException(
        status_code=400,
        detail=f"Unsupported file type: {ext or 'unknown'}",
    )


def _query_vectors(query: str, modalities: list[MediaType]) -> dict[str, list[float]]:
    """CLIP text encoder: same query vector works for text/audio/video and cross-modal image search."""
    if not modalities:
        return {}
    q = embed_text_clip(query)
    return {m.value: q for m in modalities}


def _normalize_for_match(text: str) -> str:
    cleaned = text.lower()
    cleaned = re.sub(r"\\[a-zA-Z]+", " ", cleaned)
    cleaned = re.sub(r"[^a-z0-9\s]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _query_terms(query: str) -> list[str]:
    terms = []
    for token in _normalize_for_match(query).split():
        if len(token) < 2 or token in _STOPWORDS:
            continue
        terms.append(token)
    return terms


def _term_variants(term: str) -> set[str]:
    variants = {term}
    if term.endswith("ies") and len(term) > 3:
        variants.add(term[:-3] + "y")
    if term.endswith("es") and len(term) > 3:
        variants.add(term[:-2])
    if term.endswith("s") and len(term) > 3:
        variants.add(term[:-1])
    else:
        variants.add(term + "s")
        variants.add(term + "es")
    return {v for v in variants if len(v) >= 3}


def _is_text_like_hit(hit: dict) -> bool:
    payload = hit.get("payload") or {}
    media_type = payload.get("media_type")
    if media_type in {"text", "audio"}:
        return True
    return media_type == "video" and payload.get("chunk_type") == "transcript"


def _lexical_score(query: str, hit: dict) -> float:
    terms = _query_terms(query)
    if not terms:
        return 0.0

    payload = hit.get("payload") or {}
    haystack = _normalize_for_match(
        f"{payload.get('content', '')} {payload.get('source_file', '')}"
    )
    if not haystack:
        return 0.0

    haystack_tokens = set(haystack.split())
    matched = 0
    for term in terms:
        variants = _term_variants(term)
        if any(variant in haystack_tokens for variant in variants):
            matched += 1
            continue
        if any(re.search(rf"\b{re.escape(variant)}\b", haystack) for variant in variants):
            matched += 1

    coverage = matched / len(terms)
    normalized_query = _normalize_for_match(query)
    phrase = 1.0 if re.search(rf"\b{re.escape(normalized_query)}\b", haystack) else 0.0
    return max(coverage, phrase)


def _reranked_score(query: str, hit: dict) -> float:
    raw_score = float(hit.get("score", 0.0))
    if not _is_text_like_hit(hit):
        return raw_score

    if not _query_terms(query):
        return raw_score

    lexical = _lexical_score(query, hit)
    if lexical <= 0.0:
        return raw_score * 0.1
    return (0.7 * lexical) + (0.3 * raw_score)


def _search_hits(
    query: str,
    modalities: list[MediaType],
    top_k: int,
    score_threshold: float,
) -> list[dict]:
    collections = [m.value for m in modalities]
    q_vecs = _query_vectors(query, modalities)
    grouped_hits = vector_store.search_grouped(
        collections=collections,
        query_vectors=q_vecs,
        top_k=max(top_k * 5, 10),
        score_threshold=0.0,
    )

    hits: list[dict] = []
    for col in collections:
        hits.extend(grouped_hits.get(col, []))

    for hit in hits:
        hit["raw_score"] = float(hit.get("score", 0.0))
        hit["score"] = _reranked_score(query, hit)

    hits.sort(key=lambda h: h["score"], reverse=True)
    if score_threshold > 0.0:
        hits = [h for h in hits if float(h["score"]) >= score_threshold]
    return hits[:top_k]


def _hit_to_retrieved(hit: dict) -> RetrievedChunk:
    payload = hit.get("payload") or {}
    raw_mt = payload.get("media_type", "text")
    media_type = MediaType(raw_mt) if isinstance(raw_mt, str) else raw_mt
    meta = ChunkMetadata(
        source_file=payload.get("source_file", "unknown"),
        media_type=media_type,
        chunk_index=int(payload.get("chunk_index", 0)),
        timestamp_start=payload.get("timestamp_start"),
        timestamp_end=payload.get("timestamp_end"),
        page_number=payload.get("page_number"),
        frame_index=payload.get("frame_index"),
        image_path=payload.get("image_path"),
        image_base64=payload.get("image_base64"),
        image_mime_type=payload.get("image_mime_type"),
    )
    return RetrievedChunk(
        id=str(hit["id"]),
        score=float(hit["score"]),
        content=str(payload.get("content", "")),
        metadata=meta,
    )


TaskState = Literal["PENDING", "STARTED", "SUCCESS", "FAILURE", "RETRY"]


def _normalize_task_state(raw: str) -> TaskState:
    if raw == "PROGRESS":
        return "STARTED"
    if raw in ("PENDING", "STARTED", "SUCCESS", "FAILURE", "RETRY"):
        return raw
    return "PENDING"


def _celery_status(task_id: str) -> TaskStatusResponse:
    r = AsyncResult(task_id, app=celery_app)
    raw_state = r.state
    state = _normalize_task_state(raw_state)
    progress = None
    err = None
    result = None

    if state in ("STARTED", "RETRY") and isinstance(r.info, dict):
        progress = r.info.get("progress")
    elif state == "SUCCESS":
        result = r.result if isinstance(r.result, dict) else {"value": r.result}
    elif state == "FAILURE":
        err = str(r.info) if r.info else "Task failed"

    return TaskStatusResponse(
        task_id=task_id,
        status=state,
        progress=progress,
        result=result,
        error=err,
    )


app = FastAPI(
    title="Multimodal RAG API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _ensure_upload_dir() -> None:
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)


@app.get("/api/info")
def api_info():
    return get_provider_info()


@app.post("/api/ingest", response_model=IngestResponse)
async def api_ingest(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    media_type = _detect_media_type(file.filename)
    max_bytes = settings.max_upload_mb * 1024 * 1024
    data = await file.read()
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail="File too large")

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = f"{uuid.uuid4().hex}_{Path(file.filename).name}"
    dest = upload_dir / safe_name
    dest.write_bytes(data)

    path_str = str(dest)
    name = file.filename

    if media_type == MediaType.text:
        async_result = ingest_text.delay(path_str, name)
    elif media_type == MediaType.image:
        async_result = ingest_image.delay(path_str, name)
    elif media_type == MediaType.audio:
        async_result = ingest_audio.delay(path_str, name)
    else:
        async_result = ingest_video.delay(path_str, name)

    return IngestResponse(
        task_id=async_result.id,
        filename=name,
        media_type=media_type,
        status="queued",
        message="File queued for processing",
    )


@app.get("/api/task/{task_id}", response_model=TaskStatusResponse)
def api_task(task_id: str):
    return _celery_status(task_id)


@app.post("/api/retrieve", response_model=RetrievalResponse)
def api_retrieve(body: RetrievalRequest):
    hits = _search_hits(
        query=body.query,
        modalities=body.modalities,
        top_k=body.top_k,
        score_threshold=body.score_threshold,
    )
    chunks = [_hit_to_retrieved(h) for h in hits]
    return RetrievalResponse(
        query=body.query,
        chunks=chunks,
        total_found=len(chunks),
    )


@app.post("/api/generate")
async def api_generate(body: GenerateRequest):
    hits = _search_hits(
        query=body.query,
        modalities=body.modalities,
        top_k=body.top_k,
        score_threshold=body.score_threshold,
    )
    chunks_context = hits
    info = get_provider_info()

    if body.stream:

        async def gen():
            async for token in generate_stream(
                body.query,
                chunks_context,
                system_prompt=body.system_prompt,
            ):
                yield f"data: {token}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    answer = await generate(
        body.query,
        chunks_context,
        system_prompt=body.system_prompt,
    )
    chunks_used = [_hit_to_retrieved(h) for h in hits]
    return GenerateResponse(
        answer=answer or "",
        chunks_used=chunks_used,
        model=info["model"],
        provider=info["provider"],
    )


@app.get("/health")
def health():
    return {"status": "ok"} if os.path.exists(settings.upload_dir) else {"status": "degraded"}
