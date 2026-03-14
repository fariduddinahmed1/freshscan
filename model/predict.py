import tensorflow as tf
from PIL import Image
import numpy as np

# Load model
model = tf.keras.models.load_model('model/freshscan_model.keras')

LABELS = ['fresh', 'rotten']

def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, 0)
    return img

def predict(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    confidence = float(np.max(prediction)) * 100
    label_index = np.argmax(prediction)
    label = LABELS[label_index]

    # Determine category
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

# Test it
if __name__ == "__main__":
    result = predict("test_image.jpg")
    print(result)