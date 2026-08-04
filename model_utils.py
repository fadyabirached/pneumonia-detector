"""Pure helper functions backing the Streamlit app.

Kept separate from app.py so the pre-/post-processing logic can be unit
tested without a Streamlit runtime, a real TensorFlow install, or the
~200 MB trained model file. TensorFlow is imported lazily inside the
functions that need it, which keeps `import model_utils` cheap and lets
tests stub the TF calls out via `sys.modules`.
"""
import os

import numpy as np
from PIL import Image

MODEL_PATH = "resnet50_pneumonia.keras"
CLASS_NAMES = ["Normal", "Pneumonia"]
IMAGE_SIZE = (224, 224)


def prepare_image(img: Image.Image) -> np.ndarray:
    """Convert a PIL image into a ``(1, 224, 224, 3)`` batch for the model.

    RGB conversion and resize only, no rescaling or channel normalization.
    That looks wrong for a ResNet50 (the standard advice is to run inputs
    through ``keras.applications.resnet50.preprocess_input``), but the
    training notebook's ``ImageDataGenerator`` calls (cell 10) set neither
    ``rescale`` nor ``preprocessing_function``, so the model was trained on
    raw 0-255 pixel values. Applying ``preprocess_input`` here previously
    fed it RGB-to-BGR-converted, ImageNet-mean-subtracted values instead,
    a distribution the model never saw during training. Matching the
    generators exactly, not "correct" preprocessing in the abstract, is
    what makes served predictions match the notebook's reported metrics.
    """
    img = img.convert("RGB").resize(IMAGE_SIZE)
    x = np.array(img, dtype=np.float32)
    x = np.expand_dims(x, axis=0)  # (1, 224, 224, 3)
    return x


def predicted_class(probability: float, threshold: float = 0.5) -> str:
    """Map a P(Pneumonia) score to a class label given a decision threshold."""
    index = 1 if probability >= threshold else 0
    return CLASS_NAMES[index]


def load_trained_model(path: str = MODEL_PATH):
    """Load the trained `.keras` model, failing with a clear message if missing.

    The model file is not committed to the repo (too large for git), see
    the README for how to produce it by running the training notebook.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found at '{path}'. "
            "Run the training notebook (pneumonia_detector.ipynb) to "
            "produce it, then place the .keras file in the repo root "
            "(same folder as app.py), or update MODEL_PATH."
        )
    from tensorflow.keras.models import load_model

    return load_model(path)
