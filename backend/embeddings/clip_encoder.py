from __future__ import annotations

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from config import settings

_model: CLIPModel | None = None
_processor: CLIPProcessor | None = None
_loaded_model_id: str | None = None


def _model_id() -> str:
    return settings.hf_clip_model_id


def _load() -> None:
    global _model, _processor, _loaded_model_id
    mid = _model_id()
    if _model is None or _loaded_model_id != mid:
        _model = CLIPModel.from_pretrained(mid)
        _processor = CLIPProcessor.from_pretrained(mid)
        _model.eval()
        _loaded_model_id = mid


def get_embedding_dimension() -> int:
    """Vector size for the configured CLIP model (shared text + image space)."""
    _load()
    assert _model is not None
    return int(_model.config.projection_dim)


def embed_image(image_path: str) -> list[float]:
    _load()
    assert _processor is not None and _model is not None
    image = Image.open(image_path).convert("RGB")
    inputs = _processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = _model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
    return features[0].tolist()


def embed_image_pil(image: Image.Image) -> list[float]:
    _load()
    assert _processor is not None and _model is not None
    inputs = _processor(images=image.convert("RGB"), return_tensors="pt")
    with torch.no_grad():
        features = _model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
    return features[0].tolist()


def embed_texts_clip(texts: list[str]) -> list[list[float]]:
    """Batch CLIP text embeddings (local HF model, no API)."""
    if not texts:
        return []
    _load()
    assert _processor is not None and _model is not None
    batch_size = 32
    all_rows: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = _processor(
            text=batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        with torch.no_grad():
            features = _model.get_text_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        all_rows.extend(features.tolist())
    return all_rows


def embed_text_clip(text: str) -> list[float]:
    return embed_texts_clip([text])[0]
