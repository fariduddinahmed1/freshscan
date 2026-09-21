"""YOLO fruit detection. Singleton model, same filtering as before."""
import cv2
import numpy as np
from ultralytics import YOLO

from .config import FRUIT_CLASSES, YOLO_CONF, YOLO_WEIGHTS

_model = None


def get_model(weights: str = YOLO_WEIGHTS) -> YOLO:
    global _model
    if _model is None:
        _model = YOLO(weights)
    return _model


def detect_fruits(image_bytes: bytes):
    """Returns (detections, bgr_image). Detection = [{class, confidence, bbox}]."""
    model = get_model()
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    detections = []
    for result in model(img, verbose=False):
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])
            if class_name in FRUIT_CLASSES and confidence > YOLO_CONF:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({
                    "class": class_name,
                    "confidence": round(confidence * 100, 2),
                    "bbox": [x1, y1, x2, y2],
                })
    return detections, img
