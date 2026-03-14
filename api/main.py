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

app.mount("/", StaticFiles(directory="static", html=True), name="static")