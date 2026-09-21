"""YOLO fruit detection. YOLO inference runs in a helper subprocess because
torch's lazy triton import segfaults when first loaded after tensorflow
in the same process (kills the server with no traceback)."""
import json
import subprocess
import sys

import cv2
import numpy as np
from ultralytics import YOLO

from .config import FRUIT_CLASSES, YOLO_CONF, YOLO_WEIGHTS
from .errors import invalid

_model = None

# Minimal worker: reads image bytes on stdin, prints raw boxes as JSON.
# Keeps torch/triton in a clean process (never imports tensorflow).
_WORKER_SRC = (
    "import json, sys\n"
    "import cv2\n"
    "import numpy as np\n"
    "from ultralytics import YOLO\n"
    "data = sys.stdin.buffer.read()\n"
    "img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)\n"
    "out = []\n"
    "if img is not None:\n"
    "    model = YOLO(sys.argv[1])\n"
    "    for result in model(img, verbose=False):\n"
    "        for box in result.boxes:\n"
    "            out.append({'class': model.names[int(box.cls[0])],"
    " 'confidence': float(box.conf[0]), 'bbox': list(map(int, box.xyxy[0]))})\n"
    "print(json.dumps(out))\n"
)


def get_model(weights: str = YOLO_WEIGHTS) -> YOLO:
    global _model
    if _model is None:
        _model = YOLO(weights)
    return _model


def detect_fruits(image_bytes: bytes):
    """Returns (detections, bgr_image). Detection = [{class, confidence, bbox}]."""
    if not image_bytes:
        raise invalid("Empty upload")
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise invalid("Unreadable image")
    proc = subprocess.run(
        [sys.executable, "-c", _WORKER_SRC, YOLO_WEIGHTS],
        input=image_bytes, capture_output=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError("fruit detection failed")
    try:
        raw = json.loads(proc.stdout.decode().strip().splitlines()[-1])
    except (IndexError, ValueError):
        raise RuntimeError("fruit detection failed")
    h, w = img.shape[:2]
    detections = []
    for det in raw:
        if det["class"] not in FRUIT_CLASSES or det["confidence"] <= YOLO_CONF:
            continue
        x1, y1, x2, y2 = det["bbox"]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append({
            "class": det["class"],
            "confidence": round(det["confidence"] * 100, 2),
            "bbox": [x1, y1, x2, y2],
        })
    return detections, img
