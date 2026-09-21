"""Single source of truth for paths and thresholds (env-overridable)."""
import os

MODEL_PATH = os.getenv("FRESHSCAN_MODEL", "model/freshscan_model.keras")
YOLO_WEIGHTS = os.getenv("FRESHSCAN_YOLO", "yolov8n.pt")
DB_PATH = os.getenv("FRESHSCAN_DB", "freshscan.db")

LABELS = ["fresh", "rotten"]
FRESH_THRESHOLD = 85.0
YOLO_CONF = 0.4
FRUIT_CLASSES = ("apple", "banana", "orange")
