from pydantic import BaseModel, Field
from typing import Any, Literal, Optional
from enum import Enum


class MediaType(str, Enum):
    text = "text"
    image = "image"
    audio = "audio"
    video = "video"


# --- Ingestion ---

class IngestResponse(BaseModel):
    task_id: str
    filename: str
    media_type: MediaType
    status: str = "queued"
    message: str = "File queued for processing"


class TaskStatusResponse(BaseModel):
    task_id: str
    status: Literal["PENDING", "STARTED", "SUCCESS", "FAILURE", "RETRY"]
    progress: Optional[int] = None          # 0-100
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None


# --- Retrieval ---

class RetrievalRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)
    modalities: list[MediaType] = Field(
        default=[MediaType.text, MediaType.image, MediaType.audio, MediaType.video]
    )
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0)


class ChunkMetadata(BaseModel):
    source_file: str
    media_type: MediaType
    chunk_index: int
    timestamp_start: Optional[float] = None   # for audio/video (seconds)
    timestamp_end: Optional[float] = None
    page_number: Optional[int] = None          # for PDF
    frame_index: Optional[int] = None          # for video keyframes
    image_path: Optional[str] = None
    image_base64: Optional[str] = None
    image_mime_type: Optional[str] = None


class RetrievedChunk(BaseModel):
    id: str
    score: float
    content: str                               # text or description
    metadata: ChunkMetadata


class RetrievalResponse(BaseModel):
    query: str
    chunks: list[RetrievedChunk]
    total_found: int


# --- Generation ---

class GenerateRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)
    modalities: list[MediaType] = Field(
        default=[MediaType.text, MediaType.image, MediaType.audio, MediaType.video]
    )
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    stream: bool = True
    system_prompt: Optional[str] = None


class GenerateResponse(BaseModel):
    answer: str
    chunks_used: list[RetrievedChunk]
    model: str
    provider: str
    cached: bool = False
