"""CLI inference. Reuses core/classify.py — no duplicated logic."""
import sys

import tensorflow as tf

from core.classify import classify_bytes
from core.config import MODEL_PATH

model = tf.keras.models.load_model(MODEL_PATH)


def predict(image_path: str) -> dict:
    with open(image_path, "rb") as f:
        return classify_bytes(model, f.read())


if __name__ == "__main__":
    print(predict(sys.argv[1] if len(sys.argv) > 1 else "test_image.jpg"))
