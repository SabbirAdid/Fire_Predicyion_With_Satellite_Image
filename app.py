import gradio as gr
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import random

# ✅ মডেল লোড
model = load_model("final_ensemble_fire_model.keras")
THRESHOLD = 0.6

# ✅ প্রিপ্রসেসিং ফাংশন
def preprocess_image(image, size=(350, 350), augment=False):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, size)

    if augment:
        if random.random() < 0.5:
            image = cv2.flip(image, 1)
        if random.random() < 0.3:
            image = cv2.flip(image, 0)
        if random.random() < 0.4:
            angle = random.choice([-15, -10, -5, 5, 10, 15])
            h, w = image.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
            image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        if random.random() < 0.4:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            factor = random.uniform(0.7, 1.3)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    image_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(image_clahe, -1, kernel)

    image_normalized = sharpened / 255.0
    return np.expand_dims(image_normalized, axis=0)

# ✅ TTA Prediction
def tta_prediction(image, tta_times=15):
    original = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    predictions = []
    for _ in range(tta_times):
        input_img = preprocess_image(original.copy(), augment=True)
        pred = model.predict(input_img, verbose=0)[0][0]
        predictions.append(pred)
    avg_prediction = np.mean(predictions)
    return avg_prediction, original

# ✅ Fire Prediction
def predict_fire(image):
    prediction, original_image = tta_prediction(image, tta_times=15)
    if prediction >= THRESHOLD:
        label = "🔥 Fire Detected"
    else:
        label = "✅ No Fire Detected"
    confidence = f"{prediction*100:.2f}%"
    return f"{label} | Confidence: {confidence}"

# ✅ Gradio Interface
app = gr.Interface(
    fn=predict_fire,
    inputs=gr.Image(type="pil", label="📤 Upload Image"),
    outputs=gr.Textbox(label="📊 Prediction Result"),
    title="🔥 Fire Detection App",
    description="Upload an image to detect 🔥 fire or ✅ no fire.",
    live=True
)

app.launch()
