import streamlit as st
from PIL import Image, UnidentifiedImageError

from model_utils import CLASS_NAMES, MODEL_PATH, load_trained_model, predicted_class, prepare_image

# ---------------- UI config ----------------
st.set_page_config(page_title="Pneumonia Classifier", page_icon="🫁", layout="centered")
st.title("🫁 Chest X-ray Pneumonia Classifier")
st.caption("Upload a chest X-ray image. The app returns a probability and a class prediction.")


# ---------------- Model load (cached) ----------------
@st.cache_resource
def get_model():
    return load_trained_model(MODEL_PATH)


try:
    model = get_model()
    model_ready = True
except Exception as e:
    model_ready = False
    st.error("Model could not be loaded.")
    st.exception(e)

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
        label = predicted_class(prob_pneumonia, threshold)

        st.subheader("Result")
        st.write(f"**Prediction:** {label}")
        st.write(f"**P(Pneumonia):** `{prob_pneumonia:.3f}`")
        st.write(f"**Threshold:** `{threshold:.2f}`")

        if show_debug:
            st.info(f"Input tensor shape: {x.shape}, dtype: {x.dtype}")
            st.info("Note: probability displayed is P(Pneumonia). Threshold controls the decision boundary.")
            st.info(f"Class names: {CLASS_NAMES}")

else:
    st.info("Upload an image to begin.")
