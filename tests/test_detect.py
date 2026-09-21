"""Regression test for the cv2.dnn detection path. Fails if logic breaks."""
import os

import pytest
from fastapi import HTTPException

from core.detect import detect_fruits

BANANA = os.path.join(os.path.dirname(__file__), "..",
                      "isolated-cavendish-bananas-on-white-background-it-is-a-fruit-with-good-taste-it-has-a-delicious-aroma-the-peel-is-thin-not-sticky-the-skin-color-of-bananas-turns-golden-yellow-when-ripe-photo.jpg")


def test_banana_detected():
    with open(BANANA, "rb") as f:
        detections, _ = detect_fruits(f.read())
    assert len(detections) >= 1
    assert detections[0]["class"] == "banana"
    x1, y1, x2, y2 = detections[0]["bbox"]
    assert x2 > x1 and y2 > y1


def test_garbage_rejected():
    with pytest.raises(HTTPException) as e:
        detect_fruits(b"not-an-image")
    assert e.value.status_code == 422
