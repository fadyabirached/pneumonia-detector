"""Pure helper functions backing the Streamlit app.

Kept separate from app.py so the pre-/post-processing logic can be unit
tested without a Streamlit runtime, a real TensorFlow install, or the
~200 MB trained model file. TensorFlow and huggingface_hub are both
imported lazily inside the functions that need them, which keeps
`import model_utils` cheap and lets tests stub those calls out via
`sys.modules`.
"""
import os

import numpy as np
from PIL import Image

MODEL_PATH = "resnet50_pneumonia.keras"
CLASS_NAMES = ["Normal", "Pneumonia"]
IMAGE_SIZE = (224, 224)

# Set on Streamlit Cloud (and anywhere else the ~200 MB model shouldn't be
# committed to git) to a public Hugging Face model repo, e.g.
# "yourusername/pneumonia-detector-resnet50". Left unset, load_trained_model
# behaves exactly as before: local file or a clear error.
HF_MODEL_REPO_ENV = "HF_MODEL_REPO"
HF_MODEL_FILENAME_ENV = "HF_MODEL_FILENAME"


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
    """Load the trained `.keras` model, downloading it if it isn't local.

    Checks ``path`` first (unchanged behavior for anyone who trained
    locally and dropped the file next to app.py). If it's missing and
    ``HF_MODEL_REPO`` is set, downloads it from that public Hugging Face
    model repo instead, this is what a deployed instance (Streamlit
    Cloud, a container, anywhere the 200 MB file can't live in git) uses.
    ``huggingface_hub`` caches the download locally after the first call,
    so this only touches the network once per machine.
    """
    if not os.path.exists(path):
        path = _download_from_hub(path)

    from tensorflow.keras.models import load_model

    return load_model(path)


def _download_from_hub(path: str) -> str:
    """Fetch the model from Hugging Face Hub, or raise the same clear error as before."""
    repo_id = os.environ.get(HF_MODEL_REPO_ENV)
    if not repo_id:
        raise FileNotFoundError(
            f"Model file not found at '{path}'. "
            "Run the training notebook (pneumonia_detector.ipynb) to "
            "produce it, then place the .keras file in the repo root "
            "(same folder as app.py), or set the HF_MODEL_REPO environment "
            "variable to a public Hugging Face model repo to download it "
            "automatically (see the README)."
        )

    from huggingface_hub import hf_hub_download

    filename = os.environ.get(HF_MODEL_FILENAME_ENV, os.path.basename(path))
    return hf_hub_download(repo_id=repo_id, filename=filename)
