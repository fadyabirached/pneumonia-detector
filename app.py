import os
import streamlit as st
import numpy as np
from PIL import Image, UnidentifiedImageError

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

# ---------------- UI config ----------------
st.set_page_config(page_title="Pneumonia Classifier", page_icon="🫁", layout="centered")
st.title("🫁 Chest X-ray Pneumonia Classifier")
st.caption("Upload a chest X-ray image. The app returns a probability and a class prediction.")

MODEL_PATH = "resnet50_pneumonia.keras"
CLASS_NAMES = ["Normal", "Pneumonia"]

# ---------------- Model load (cached) ----------------
@st.cache_resource
def load_trained_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found at '{path}'. "
            "Make sure you placed the .keras file in the repo root (same folder as app.py) "
            "or update MODEL_PATH."
        )
    return load_model(path)

try:
    model = load_trained_model(MODEL_PATH)
    model_ready = True
except Exception as e:
    model_ready = False
    st.error("Model could not be loaded.")
    st.exception(e)

# ---------------- Helpers ----------------
def prepare_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((224, 224))
    x = np.array(img, dtype=np.float32)
    x = np.expand_dims(x, axis=0)          # (1,224,224,3)
    x = preprocess_input(x)
    return x

# ---------------- Sidebar controls ----------------
st.sidebar.header("Settings")
threshold = st.sidebar.slider("Decision threshold", 0.05, 0.95, 0.50, 0.01)
show_debug = st.sidebar.checkbox("Show debug info", value=False)

# ---------------- Main app ----------------
file = st.file_uploader("Upload a chest X-ray (JPG/PNG)", type=["jpg", "jpeg", "png"])

if file is not None:
    try:
        img = Image.open(file)
    except UnidentifiedImageError:
        st.error("This file doesn't look like a valid image. Please upload a JPG/PNG.")
        st.stop()

    st.image(img, caption="Input X-ray", use_container_width=True)

    if not model_ready:
        st.warning("Upload works, but prediction is disabled until the model loads correctly.")
        st.stop()

    if st.button("Predict", type="primary"):
        x = prepare_image(img)
        prob_pneumonia = float(model.predict(x, verbose=0).ravel()[0])
        pred = 1 if prob_pneumonia >= threshold else 0

        st.subheader("Result")
        st.write(f"**Prediction:** {CLASS_NAMES[pred]}")
        st.write(f"**P(Pneumonia):** `{prob_pneumonia:.3f}`")
        st.write(f"**Threshold:** `{threshold:.2f}`")

        if show_debug:
            st.info(f"Input tensor shape: {x.shape}, dtype: {x.dtype}")
            st.info("Note: probability displayed is P(Pneumonia). Threshold controls the decision boundary.")

else:
    st.info("Upload an image to begin.")
