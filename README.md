# FreshScan 🥦

> AI-powered food freshness detection system using dual deep learning models.

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-latest-green?style=flat-square)
![YOLOv8](https://img.shields.io/badge/YOLOv8-ultralytics-red?style=flat-square)
![Accuracy](https://img.shields.io/badge/Accuracy-99.4%25-brightgreen?style=flat-square)

---

## What it does

FreshScan uses two AI models working together to detect whether food is fresh, should be consumed soon, or should be discarded — in real time via webcam.

- **MobileNetV2** classifies freshness with 99.4% accuracy
- **YOLOv8** detects and localizes fruits with bounding boxes
- **Shelf tracker** monitors inventory and sends email alerts when items degrade
- **Live camera** auto-scans every 2 seconds, Google Lens style

---

## Demo

| Scanner Tab | YOLOv8 Detect | Shelf Tracker |
|-------------|---------------|---------------|
| Live camera feed with auto-scan | Real-time bounding boxes | Track items by shelf & box |
| Fresh / Consume Soon / Rotten | Apple, Banana, Orange | Email alerts on degradation |

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| ML Model | TensorFlow + MobileNetV2 (transfer learning) |
| Object Detection | YOLOv8n (Ultralytics) |
| Backend | FastAPI + Uvicorn |
| Frontend | HTML / CSS / JS (single file) |
| Notifications | Gmail SMTP |
| Dataset | Kaggle — Fruits Fresh and Rotten (10,901 train images) |

---

## Model Performance

- **Architecture**: MobileNetV2 (frozen base) + GlobalAveragePooling2D + Dense(128) + Dense(2)
- **Training**: 5 epochs on Google Colab T4 GPU
- **Val Accuracy**: 99.44%
- **Dataset**: 4,740 fresh + 6,161 rotten images (train), 2,698 test images
- **Classes**: Fresh → Consume Soon → Rotten (3-tier classification)

---

## Project Structure

```
freshscan/
├── model/
│   ├── train.py          # MobileNetV2 training script
│   ├── predict.py        # Inference pipeline
│   └── freshscan_model.keras  # Trained model (Google Drive)
├── api/
│   └── main.py           # FastAPI backend
├── static/
│   └── index.html        # Frontend UI
├── requirements.txt
└── README.md
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Classify freshness from image |
| POST | `/detect` | YOLOv8 detection + classification + annotated image |
| POST | `/shelf/add` | Add item to shelf tracker |
| POST | `/shelf/scan/{shelf}/{box}` | Scan a tracked item |
| GET | `/shelf/all` | List all tracked items |
| DELETE | `/shelf/{shelf}/{box}` | Remove tracked item |
| POST | `/config/email` | Configure Gmail SMTP alerts |
| GET | `/health` | Health check |
| GET | `/docs` | Swagger UI |

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/Bappaditya-kuilya/freshscan.git
cd freshscan
```

### 2. Create virtual environment

```bash
uv venv
.venv\Scripts\activate  # Windows
```

### 3. Install dependencies

```bash
uv pip install -r requirements.txt
```

### 4. Add the trained model

Download `freshscan_model.keras` from Google Drive and place it in `model/`.

### 5. Run the server

```bash
uvicorn api.main:app --reload
```

Visit `http://127.0.0.1:8000`

---

## Features

- **Real-time scanning** — auto-analyzes webcam feed every 2 seconds
- **3-tier classification** — Fresh / Consume Soon / Rotten with confidence %
- **Shelf life estimate** — tells you how long the item will last
- **YOLOv8 detection** — bounding boxes around detected fruits
- **Scan history** — last 6 scans stored in session
- **Shelf tracker** — add items by shelf/box number
- **Email alerts** — Gmail notifications when items degrade
- **Swagger docs** — full API documentation at `/docs`

---

## Built by

Bappaditya-kuilya.



---

## License

MIT