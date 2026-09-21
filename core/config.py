"""Single source of truth for paths and thresholds (env-overridable)."""
import os

MODEL_PATH = os.getenv("FRESHSCAN_MODEL", "model/freshscan_model.keras")
YOLO_MODEL = os.getenv("FRESHSCAN_YOLO", "model/yolov8n.onnx")
DB_PATH = os.path.abspath(os.getenv("FRESHSCAN_DB", "freshscan.db"))

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH} (set FRESHSCAN_MODEL env var)"
    )

LABELS = ["fresh", "rotten"]
FRESH_THRESHOLD = 85.0
YOLO_CONF = 0.4
FRUIT_CLASSES = ("apple", "banana", "orange")
