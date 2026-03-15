from ultralytics import YOLO

# Load YOLOv8
yolo_model = YOLO('yolov8n.pt')
FRUIT_CLASSES = ['apple', 'banana', 'orange']

def detect_fruits(image_bytes):
    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    results = yolo_model(img)
    detections = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = yolo_model.names[class_id]
            confidence = float(box.conf[0])
            if class_name in FRUIT_CLASSES and confidence > 0.4:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({
                    "class": class_name,
                    "confidence": round(confidence * 100, 2),
                    "bbox": [x1, y1, x2, y2]
                })
    return detections, img
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Optional
import json
import os
import cv2
import base64

app = FastAPI(title="FreshScan API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
model = tf.keras.models.load_model('model/freshscan_model.keras')
LABELS = ['fresh', 'rotten']

# In-memory storage for shelf tracking
shelf_data = {}

# Email config
EMAIL_CONFIG = {
    "sender": "",
    "password": "",
    "recipient": ""
}

# ─── Models ───────────────────────────────────────────

class ShelfItem(BaseModel):
    shelf_number: str
    box_number: str
    item_name: str
    email: Optional[str] = None

class EmailConfig(BaseModel):
    sender_email: str
    sender_password: str
    recipient_email: str

# ─── Helper Functions ──────────────────────────────────

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    img = img.resize((224, 224))
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, 0)
    return img

def send_email_alert(recipient, item_name, shelf, box, category, shelf_life):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_CONFIG['sender']
        msg['To'] = recipient
        msg['Subject'] = f"FreshScan Alert — {item_name} needs attention!"

        body = f"""
        FreshScan Alert 🚨

        Item: {item_name}
        Location: Shelf {shelf}, Box {box}
        Status: {category}
        Shelf Life: {shelf_life}
        Detected at: {datetime.now().strftime('%Y-%m-%d %H:%M')}

        Please take action immediately.

        — FreshScan AI System
        """

        msg.attach(MIMEText(body, 'plain'))
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_CONFIG['sender'], EMAIL_CONFIG['password'])
        server.sendmail(EMAIL_CONFIG['sender'], recipient, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        print(f"Email error: {e}")
        return False

# ─── Endpoints ────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "FreshScan is running!"}

@app.post("/config/email")
def configure_email(config: EmailConfig):
    EMAIL_CONFIG['sender'] = config.sender_email
    EMAIL_CONFIG['password'] = config.sender_password
    EMAIL_CONFIG['recipient'] = config.recipient_email
    return {"status": "Email configured successfully"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = preprocess_image(contents)
    prediction = model.predict(img)
    confidence = float(np.max(prediction)) * 100
    label = LABELS[np.argmax(prediction)]

    if label == 'fresh' and confidence >= 85:
        category = 'Fresh'
        shelf_life = f"{int(confidence / 20) + 2} days"
    elif label == 'fresh' and confidence < 85:
        category = 'Consume Soon'
        shelf_life = "1-2 days"
    else:
        category = 'Rotten'
        shelf_life = "Discard immediately"

    return {
        "category": category,
        "confidence": round(confidence, 2),
        "shelf_life": shelf_life,
        "timestamp": datetime.now().strftime('%H:%M:%S')
    }

@app.post("/shelf/add")
def add_shelf_item(item: ShelfItem):
    key = f"{item.shelf_number}_{item.box_number}"
    shelf_data[key] = {
        "shelf_number": item.shelf_number,
        "box_number": item.box_number,
        "item_name": item.item_name,
        "email": item.email,
        "scans": [],
        "added_at": datetime.now().strftime('%Y-%m-%d %H:%M')
    }
    return {"status": "Item added", "key": key}

@app.post("/shelf/scan/{shelf_number}/{box_number}")
async def scan_shelf_item(shelf_number: str, box_number: str, file: UploadFile = File(...)):
    key = f"{shelf_number}_{box_number}"

    contents = await file.read()
    img = preprocess_image(contents)
    prediction = model.predict(img)
    confidence = float(np.max(prediction)) * 100
    label = LABELS[np.argmax(prediction)]

    if label == 'fresh' and confidence >= 85:
        category = 'Fresh'
        shelf_life = f"{int(confidence / 20) + 2} days"
    elif label == 'fresh' and confidence < 85:
        category = 'Consume Soon'
        shelf_life = "1-2 days"
    else:
        category = 'Rotten'
        shelf_life = "Discard immediately"

    result = {
        "category": category,
        "confidence": round(confidence, 2),
        "shelf_life": shelf_life,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M')
    }

    # Save scan to shelf data
    if key in shelf_data:
        shelf_data[key]['scans'].append(result)
        shelf_data[key]['latest'] = result

        # Send email if Consume Soon or Rotten
        if category in ['Consume Soon', 'Rotten']:
            email = shelf_data[key].get('email') or EMAIL_CONFIG.get('recipient')
            if email and EMAIL_CONFIG['sender']:
                send_email_alert(
                    email,
                    shelf_data[key]['item_name'],
                    shelf_number,
                    box_number,
                    category,
                    shelf_life
                )

    return result

@app.get("/shelf/all")
def get_all_shelf_items():
    return shelf_data

@app.delete("/shelf/{shelf_number}/{box_number}")
def delete_shelf_item(shelf_number: str, box_number: str):
    key = f"{shelf_number}_{box_number}"
    if key in shelf_data:
        del shelf_data[key]
        return {"status": "Deleted"}
    return {"status": "Not found"}



@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    
    # Step 1 - YOLOv8 detects fruits and locations
    detections, img = detect_fruits(contents)
    
    results = []
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        
        # Step 2 - Crop detected fruit
        cropped = img[y1:y2, x1:x2]
        
        # Step 3 - Convert crop to bytes for MobileNetV2
        _, buffer = cv2.imencode('.jpg', cropped)
        crop_bytes = buffer.tobytes()
        
        # Step 4 - MobileNetV2 classifies the crop
        processed = preprocess_image(crop_bytes)
        prediction = model.predict(processed)
        confidence = float(np.max(prediction)) * 100
        label = LABELS[np.argmax(prediction)]

        if label == 'fresh' and confidence >= 85:
            category = 'Fresh'
            shelf_life = f"{int(confidence / 20) + 2} days"
        elif label == 'fresh' and confidence < 85:
            category = 'Consume Soon'
            shelf_life = "1-2 days"
        else:
            category = 'Rotten'
            shelf_life = "Discard immediately"

        # Step 5 - Draw box on image
        color = (0, 255, 0) if category == 'Fresh' else (0, 0, 255) if category == 'Rotten' else (0, 165, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{det['class']} - {category} {confidence:.0f}%",
                   (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        results.append({
            "fruit": det['class'],
            "category": category,
            "confidence": round(confidence, 2),
            "shelf_life": shelf_life,
            "bbox": det['bbox']
        })

    # Step 6 - Encode annotated image to base64
    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "detections": results,
        "annotated_image": img_base64,
        "count": len(results)
    }

app.mount("/", StaticFiles(directory="static", html=True), name="static")