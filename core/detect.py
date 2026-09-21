"""YOLO fruit detection via cv2.dnn on an ONNX export. No torch, ever:
torch's lazy init segfaults in a TF-owned process, so ultralytics is
export-time only (see model/README note). Same return contract as before."""
import cv2
import numpy as np

from .config import FRUIT_CLASSES, YOLO_CONF, YOLO_MODEL
from .errors import invalid

INPUT_SIZE = 640
NMS_THRESHOLD = 0.45
# COCO ids for the fruits we track.
CLASS_IDS = {"banana": 46, "apple": 47, "orange": 49}
ID_TO_FRUIT = {v: k for k, v in CLASS_IDS.items() if k in FRUIT_CLASSES}

_net = None


def get_net(path: str = YOLO_MODEL) -> cv2.dnn.Net:
    global _net
    if _net is None:
        _net = cv2.dnn.readNetFromONNX(path)
    return _net


def warmup() -> None:
    get_net()


def detect_fruits(image_bytes: bytes):
    """Returns (detections, bgr_image). Detection = [{class, confidence, bbox}]."""
    if not image_bytes:
        raise invalid("Empty upload")
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise invalid("Unreadable image")
    h, w = img.shape[:2]

    net = get_net()
    scale = min(INPUT_SIZE / w, INPUT_SIZE / h)
    nw, nh = int(w * scale), int(h * scale)
    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
    canvas[(INPUT_SIZE - nh) // 2:(INPUT_SIZE - nh) // 2 + nh,
           (INPUT_SIZE - nw) // 2:(INPUT_SIZE - nw) // 2 + nw] = cv2.resize(img, (nw, nh))
    net.setInput(cv2.dnn.blobFromImage(canvas, 1 / 255.0, swapRB=True, crop=False))
    rows = net.forward().squeeze().T  # (8400, 4 + 80)

    boxes, confs, class_ids = [], [], []
    for row in rows:
        scores = row[4:]
        class_id = int(np.argmax(scores))
        if class_id not in ID_TO_FRUIT:
            continue
        conf = float(scores[class_id])
        if conf <= YOLO_CONF:
            continue
        cx, cy, bw, bh = row[:4]
        x1 = int(((cx - bw / 2) - (INPUT_SIZE - nw) // 2) / scale)
        y1 = int(((cy - bh / 2) - (INPUT_SIZE - nh) // 2) / scale)
        boxes.append([x1, y1, int(bw / scale), int(bh / scale)])
        confs.append(conf)
        class_ids.append(class_id)

    detections = []
    if not boxes:
        return detections, img
    for i in cv2.dnn.NMSBoxes(boxes, confs, YOLO_CONF, NMS_THRESHOLD).flatten():
        x, y, bw, bh = boxes[i]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w, x + bw), min(h, y + bh)
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append({
            "class": ID_TO_FRUIT[class_ids[i]],
            "confidence": round(confs[i] * 100, 2),
            "bbox": [x1, y1, x2, y2],
        })
    return detections, img
