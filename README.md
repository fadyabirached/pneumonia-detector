# 🫁 Pneumonia Detector

[![CI](https://github.com/fadyabirached/pneumonia-detector/actions/workflows/ci.yml/badge.svg)](https://github.com/fadyabirached/pneumonia-detector/actions/workflows/ci.yml)

A **ResNet-50** chest X-ray classifier served through a **Streamlit** app.  
Fine-tuned on the [Coronahack Chest X-Ray dataset](https://www.kaggle.com/datasets/praveengovi/coronahack-chest-xraydataset) — achieving **~95.7% test accuracy** and **~0.99 ROC-AUC**.

---

## Architecture

```
Input X-ray (224×224 RGB, raw pixels)
    │
    ▼
 ResNet-50 backbone ──── ImageNet weights, frozen (stage 1) / last 30 layers unfrozen (stage 2)
    │
    ▼
 GlobalAveragePooling2D
    │
    ▼
 BatchNorm → Dense(256, ReLU) → Dropout(0.5)
    │
    ▼
 Dense(1, sigmoid) ─────── P(Pneumonia)
    │
    ▼
 Streamlit UI ─────────── Prediction + probability + adjustable threshold
```

## Results

| Metric | Score |
|---|---|
| Test Accuracy | **~95.7%** |
| ROC-AUC | **~0.99** |
| Recall (Pneumonia) | **0.95** |
| Confusion Matrix | TN 130 · FP 4 · FN 19 · TP 373 |

---

![App screenshot](https://github.com/user-attachments/assets/07b6d1be-c822-463a-b140-d25b28e679f2)

---

## Project Structure

```
pneumonia-detector/
├── app.py                        # Streamlit app (UI + inference)
├── model_utils.py                # Preprocessing & model loading — unit tested
├── pneumonia_detector.ipynb      # Data prep, two-stage training & evaluation notebook
├── requirements.txt              # Pinned dependencies
├── ruff.toml                     # Lint config (notebook excluded)
├── pyproject.toml                # Pytest config
├── tests/                        # Unit tests for model_utils.py
├── .github/workflows/ci.yml      # Lint + test on every push
├── LICENSE
├── README.md                     # This file
└── resnet50_pneumonia.keras      # Trained model — NOT in this repo, see below
```

> **`resnet50_pneumonia.keras` (~200 MB) is not committed to this repository.**
> It's too large for a normal git repo and for Streamlit Cloud's git-based deploy.
> `model_utils.load_trained_model` looks for it in the project root first; if missing,
> it downloads it from Hugging Face Hub instead (see below). Generate it yourself with
> the training notebook, or point at your own Hugging Face model repo.

---

## Quickstart

There are two separate paths depending on what you want to do:

- **"I just want to see the app run"** → you need a model file, since none ships in
  the repo. Either train your own (see **[Reproducing the model](#reproducing-the-model)**),
  or point the app at a public Hugging Face model repo (below) — no training required if
  you already have one.
- **"I already have `resnet50_pneumonia.keras`"** → follow the steps below directly.

### 1. Clone & enter the project
```bash
git clone https://github.com/fadyabirached/pneumonia-detector.git
cd pneumonia-detector
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Provide the model
Either place `resnet50_pneumonia.keras` in the project root, **or** set:
```bash
export HF_MODEL_REPO="fadyabirached/pneumonia-detector-resnet50"   # a public HF model repo
export HF_MODEL_FILENAME="resnet50_pneumonia.keras"                # optional, this is the default
```
The first call downloads and caches it via `huggingface_hub`, so this only touches the
network once per machine, not once per request.

### 4. Run
```bash
streamlit run app.py
```

---

## Reproducing the model

The notebook is written for **Google Colab**, not local Jupyter — the first data cell
calls `google.colab.files.upload()`, which only exists in Colab. Running it in local
Jupyter fails at that cell with `ModuleNotFoundError: No module named 'google'` unless
you replace that cell with a plain file path.

### What you need
- A free [Kaggle account](https://www.kaggle.com) with an API token: **Kaggle → your
  profile → Account → Create New API Token** (downloads `kaggle.json`).
- No local dataset download needed — the notebook pulls the
  [Coronahack Chest X-Ray dataset](https://www.kaggle.com/datasets/praveengovi/coronahack-chest-xraydataset)
  live from Kaggle.

### Steps
1. Open `pneumonia_detector.ipynb` in Colab.
2. Run the `files.upload()` cell and select your `kaggle.json` — this authenticates the
   next cell's `kaggle datasets download`.
3. Run the cells in order: stratified 80/10/10 split, on-the-fly augmentation
   (`brightness_range=[0.7, 1.3]`, `zoom_range=0.2`, horizontal flip), and inverse-frequency
   class weights to penalize misclassifying the rarer class (Normal).
4. Stage 1 training fine-tunes only the custom head with the ResNet-50 backbone frozen
   (`Adam(lr=1e-4)`, early stopping, `ReduceLROnPlateau`).
5. Stage 2 unfreezes the last 30 convolutional layers and continues training at a lower
   learning rate (`Adam(lr=1e-5)`) with the same callbacks.
6. The evaluation cell reports accuracy, ROC-AUC, and a confusion matrix on the 10%
   hold-out test split.
7. The final cell exports `resnet50_pneumonia.keras` (~200 MB), downloadable directly or
   as a zip.

Training end-to-end (dataset download + both fine-tuning stages) takes a while on Colab's
free-tier GPU — budget at least 30–60 minutes.

### Deploying your own live demo (Streamlit Community Cloud)
Once the model is uploaded to a public Hugging Face model repo:
1. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, click
   **New app**.
2. Pick this repo, branch `main`, main file path `app.py`.
3. Under **Advanced settings → Secrets**, add:
   ```toml
   HF_MODEL_REPO = "fadyabirached/pneumonia-detector-resnet50"
   ```
4. Click **Deploy**. First load downloads and caches the model (~200 MB), so budget a
   minute or two before the app is responsive.

---

## Testing & CI

`tests/` covers the preprocessing and model-loading logic in `model_utils.py`, without
requiring the trained model file or the dataset.

```bash
pip install pytest ruff
pytest -v                    # run the test suite
ruff check .                 # lint (notebook excluded, see ruff.toml)
```

GitHub Actions (`.github/workflows/ci.yml`) runs both on every push/PR to `main`. CI never
downloads the dataset or the trained model — it only exercises the model-free code paths.

---

## Why raw pixels, not rescaled input

`ImageDataGenerator` sets neither `rescale` nor `preprocessing_function`, so the model
trains on raw 0–255 pixel values instead of the more common 0–1 normalization. This is
intentional and matched exactly at inference: `model_utils.prepare_image` performs the
same 224×224 RGB resize with no rescaling, so training and serving stay consistent. Mixing
a rescaled training pipeline with an unrescaled inference path (or vice versa) is a classic
silent bug in image classifiers — the model still runs, it just sees inputs on the wrong
scale and produces degraded predictions.

---

## Pinned Keras version

`keras==3.15.1` is pinned explicitly in `requirements.txt` rather than left to
`tensorflow-cpu`'s own looser dependency bound. A `.keras` file saved under one Keras 3
point-release and loaded under a different one can hit a real deserialization bug where a
pretrained backbone nested inside a `Sequential` model reconstructs with a duplicated input
edge (`BatchNormalization expects 1 input but received 2`). The training notebook pins the
same exact version, so training and serving are always on identical Keras.

---

## Tech Stack

| Component | Library |
|---|---|
| Backbone | `ResNet-50` (ImageNet weights) |
| Deep learning | `TensorFlow` / `Keras` |
| Model hosting | `huggingface_hub` |
| UI | `Streamlit` |
| Image processing | `Pillow` |
| Testing / CI | `pytest`, `ruff`, GitHub Actions |

---

## License
See [LICENSE](LICENSE).
