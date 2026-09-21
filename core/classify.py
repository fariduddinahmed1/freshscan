"""Freshness classification. One shared implementation for all endpoints."""
import io

import numpy as np
from PIL import Image

from .config import FRESH_THRESHOLD, LABELS


def preprocess_image(image_bytes: bytes, size: int = 224) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((size, size))
    return np.expand_dims(np.asarray(img) / 255.0, 0)


def categorize(label: str, confidence: float) -> tuple[str, str]:
    """Pure mapping: model output -> (category, shelf life). No I/O, fully tested."""
    if label == "fresh" and confidence >= FRESH_THRESHOLD:
        return "Fresh", f"{int(confidence / 20) + 2} days"
    if label == "fresh":
        return "Consume Soon", "1-2 days"
    return "Rotten", "Discard immediately"


def classify_batch(model, img_batch: np.ndarray) -> dict:
    prediction = model.predict(img_batch, verbose=0)
    confidence = float(np.max(prediction)) * 100
    label = LABELS[int(np.argmax(prediction))]
    category, shelf_life = categorize(label, confidence)
    return {"category": category, "confidence": round(confidence, 2), "shelf_life": shelf_life}


def classify_bytes(model, image_bytes: bytes) -> dict:
    return classify_batch(model, preprocess_image(image_bytes))
