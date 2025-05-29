import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

# ▸ cache so it loads only once
@st.cache_resource
def load():
    return load_model("resnet50_pneumonia.keras")

model = load()
CLASS_NAMES = ["Normal", "Pneumonia"]

st.title("Chest-X-ray Pneumonia Classifier")

file = st.file_uploader("Upload a chest X-ray (JPEG/PNG)", type=['jpg','jpeg','png'])
if file:
    img = Image.open(file).convert("RGB").resize((224,224))
    st.image(img, caption="Input X-ray", width=300)

    if st.button("Predict"):
        x = np.expand_dims(img, 0)        # to batch ⟨1,224,224,3⟩
        x = preprocess_input(x.astype(np.float32))
        prob = float(model.predict(x, verbose=0)[0][0])
        pred = 1 if prob > 0.5 else 0
        st.write(f"### Prediction: **{CLASS_NAMES[pred]}**")
        st.write(f"Confidence: `{prob:.3f}`")
