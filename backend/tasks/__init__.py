from celery import Celery
from config import settings

celery_app = Celery(
    "rag_worker",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    result_expires=3600,
)

# Import tasks so Celery discovers them
from tasks.ingest_tasks import ingest_text, ingest_image, ingest_audio, ingest_video  # noqa

__all__ = ["celery_app"]