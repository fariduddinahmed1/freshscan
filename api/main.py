"""FreshScan API. Thin routes only — all logic lives in core/."""
import base64
import logging
from contextlib import asynccontextmanager
from datetime import datetime

import cv2
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.schemas import EmailConfig, ShelfItem
from core.classify import classify_batch, classify_bytes, preprocess_image
from core.config import MODEL_PATH
from core.detect import detect_fruits
from core.errors import read_upload_bytes
from core.mail import send_alert
from core.shelf import ShelfStore

EMAIL_CONFIG = {"sender": "", "password": "", "recipient": ""}
state: dict = {}


def _require_model():
    if "model" not in state:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return state["model"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state["model"] = tf.keras.models.load_model(MODEL_PATH)
    except Exception:
        logging.exception("Failed to load model: %s", MODEL_PATH)
        raise
    from core.detect import warmup
    warmup()
    state["shelf"] = ShelfStore()
    yield


app = FastAPI(title="FreshScan API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health():
    if "model" not in state or "shelf" not in state:
        raise HTTPException(status_code=503, detail="Not ready")
    return {"status": "FreshScan is running!"}


@app.post("/config/email")
def configure_email(config: EmailConfig):
    EMAIL_CONFIG.update(sender=config.sender_email,
                        password=config.sender_password,
                        recipient=config.recipient_email)
    return {"status": "Email configured successfully"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    model = _require_model()
    result = classify_bytes(model, await read_upload_bytes(file))
    result["timestamp"] = datetime.now().strftime("%H:%M:%S")
    return result


@app.post("/shelf/add")
def add_shelf_item(item: ShelfItem):
    key = state["shelf"].add(item.shelf_number, item.box_number, item.item_name, item.email)
    return {"status": "Item added", "key": key}


@app.post("/shelf/scan/{shelf_number}/{box_number}")
async def scan_shelf_item(shelf_number: str, box_number: str, file: UploadFile = File(...)):
    model = _require_model()
    result = classify_bytes(model, await read_upload_bytes(file))
    result["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    email_sent = False
    items = state["shelf"].get_all()
    if (item := items.get(f"{shelf_number}_{box_number}")) is not None:
        state["shelf"].record_scan(shelf_number, box_number, result)
        if result["category"] in ("Consume Soon", "Rotten"):
            email = item.get("email") or EMAIL_CONFIG.get("recipient")
            if email and EMAIL_CONFIG["sender"]:
                email_sent = send_alert(EMAIL_CONFIG["sender"], EMAIL_CONFIG["password"],
                                        email, item["item_name"],
                                        shelf_number, box_number,
                                        result["category"], result["shelf_life"])
    result["email_sent"] = email_sent
    return result


@app.get("/shelf/all")
def get_all_shelf_items():
    return state["shelf"].get_all()


@app.delete("/shelf/{shelf_number}/{box_number}")
def delete_shelf_item(shelf_number: str, box_number: str):
    if state["shelf"].delete(shelf_number, box_number):
        return {"status": "Deleted"}
    return {"status": "Not found"}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    model = _require_model()
    detections, img = detect_fruits(await read_upload_bytes(file))
    results = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        if x2 <= x1 or y2 <= y1:
            continue
        cropped = img[y1:y2, x1:x2]
        ok, buffer = cv2.imencode(".jpg", cropped)
        if not ok:
            continue
        result = classify_batch(model, preprocess_image(buffer.tobytes()))
        color = (0, 255, 0) if result["category"] == "Fresh" \
            else (0, 0, 255) if result["category"] == "Rotten" else (0, 165, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{det['class']} - {result['category']} {result['confidence']:.0f}%",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        results.append({"fruit": det["class"], **result, "bbox": det["bbox"]})
    ok, buffer = cv2.imencode(".jpg", img)
    if not ok:
        raise RuntimeError("failed to encode result image")
    return {"detections": results,
            "annotated_image": base64.b64encode(buffer).decode("utf-8"),
            "count": len(results)}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
