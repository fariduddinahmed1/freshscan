from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from PIL import Image
import numpy as np
import io

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

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    img = img.resize((224, 224))
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, 0)
    return img

@app.get("/health")
def health():
    return {"status": "FreshScan is running!"}

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
        "shelf_life": shelf_life
    }

app.mount("/", StaticFiles(directory="static", html=True), name="static")