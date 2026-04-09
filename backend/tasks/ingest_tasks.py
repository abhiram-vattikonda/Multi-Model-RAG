import os
from celery import Task
from tasks.celery_app import celery_app
from ingestion.text_ingestor import TextIngestor
from ingestion.image_ingestor import ImageIngestor
from ingestion.audio_ingestor import AudioIngestor
from ingestion.video_ingestor import VideoIngestor


class BaseIngestTask(Task):
    """Base task with shared state updates."""
    abstract = True

    def update_progress(self, current: int, total: int, msg: str = ""):
        pct = int((current / total) * 100) if total else 0
        self.update_state(
            state="STARTED",
            meta={"progress": pct, "message": msg},
        )


@celery_app.task(bind=True, base=BaseIngestTask, name="tasks.ingest_text")
def ingest_text(self, file_path: str, filename: str) -> dict:
    try:
        self.update_progress(0, 3, "Reading file...")
        ingestor = TextIngestor()
        chunks = ingestor.extract_chunks(file_path, filename)

        self.update_progress(1, 3, f"Embedding {len(chunks)} chunks...")
        vectors = ingestor.embed_chunks(chunks)

        self.update_progress(2, 3, "Storing in vector DB...")
        ids = ingestor.store(chunks, vectors)

        return {"status": "success", "chunks_stored": len(ids), "filename": filename}
    except Exception as exc:
        self.update_state(state="FAILURE", meta={"error": str(exc)})
        raise exc
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@celery_app.task(bind=True, base=BaseIngestTask, name="tasks.ingest_image")
def ingest_image(self, file_path: str, filename: str) -> dict:
    try:
        self.update_progress(0, 3, "Loading image...")
        ingestor = ImageIngestor()
        chunks = ingestor.extract_chunks(file_path, filename)

        self.update_progress(1, 3, "Encoding with CLIP...")
        vectors = ingestor.embed_chunks(chunks)

        self.update_progress(2, 3, "Storing in vector DB...")
        ids = ingestor.store(chunks, vectors)

        return {"status": "success", "chunks_stored": len(ids), "filename": filename}
    except Exception as exc:
        self.update_state(state="FAILURE", meta={"error": str(exc)})
        raise exc
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@celery_app.task(bind=True, base=BaseIngestTask, name="tasks.ingest_audio")
def ingest_audio(self, file_path: str, filename: str) -> dict:
    try:
        self.update_progress(0, 4, "Loading audio...")
        ingestor = AudioIngestor()

        self.update_progress(1, 4, "Transcribing with Whisper (this may take a while)...")
        chunks = ingestor.extract_chunks(file_path, filename)

        self.update_progress(2, 4, f"Embedding {len(chunks)} transcript chunks...")
        vectors = ingestor.embed_chunks(chunks)

        self.update_progress(3, 4, "Storing in vector DB...")
        ids = ingestor.store(chunks, vectors)

        return {"status": "success", "chunks_stored": len(ids), "filename": filename}
    except Exception as exc:
        self.update_state(state="FAILURE", meta={"error": str(exc)})
        raise exc
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@celery_app.task(bind=True, base=BaseIngestTask, name="tasks.ingest_video")
def ingest_video(self, file_path: str, filename: str) -> dict:
    try:
        self.update_progress(0, 5, "Loading video...")
        ingestor = VideoIngestor()

        self.update_progress(1, 5, "Extracting audio track...")
        audio_path = ingestor.extract_audio(file_path)

        self.update_progress(2, 5, "Transcribing with Whisper...")
        transcript_chunks = ingestor.transcribe(audio_path, filename)

        self.update_progress(3, 5, "Extracting and encoding keyframes with CLIP...")
        frame_chunks = ingestor.extract_keyframes(file_path, filename)

        all_chunks = transcript_chunks + frame_chunks
        self.update_progress(4, 5, f"Embedding & storing {len(all_chunks)} chunks...")
        vectors = ingestor.embed_chunks(all_chunks)
        ids = ingestor.store(all_chunks, vectors)

        return {
            "status": "success",
            "chunks_stored": len(ids),
            "transcript_chunks": len(transcript_chunks),
            "frame_chunks": len(frame_chunks),
            "filename": filename,
        }
    except Exception as exc:
        self.update_state(state="FAILURE", meta={"error": str(exc)})
        raise exc
    finally:
        for path in [file_path]:
            if path and os.path.exists(path):
                os.remove(path)