# Pneumonia Detector (ResNet-50)

A Streamlit web app that classifies chest X-ray images as **Normal** or **Pneumonia**, powered by a ResNet-50 model fine-tuned on the [Coronahack Chest X-Ray dataset](https://www.kaggle.com/datasets/praveengovi/coronahack-chest-xraydataset).

![App screenshot](https://github.com/user-attachments/assets/07b6d1be-c822-463a-b140-d25b28e679f2)

## Model pipeline

| Phase | What I did | Details |
|---|---|---|
| 1. Data split | Stratified 80 / 10 / 10 split into train / validation / test | Preserves the Normal / Pneumonia ratio in every subset for fair metrics |
| 2. Pre-processing | 224 x 224 RGB resize, no rescaling | `ImageDataGenerator` sets neither `rescale` nor `preprocessing_function`, so the model trains on raw 0-255 pixel values; `model_utils.prepare_image` matches this exactly at inference |
| 3. Data augmentation | `ImageDataGenerator` on the fly | `brightness_range=[0.7, 1.3]`, `zoom_range=0.2`, `horizontal_flip=True` |
| 4. Class weights | Inverse-frequency `{0: w_normal, 1: w_pneumonia}` | Penalizes misclassifying the rarer class (Normal) |
| 5. Architecture | ResNet-50 backbone (ImageNet weights, frozen) + custom head | `GlobalAveragePooling2D -> BatchNorm -> Dense(256, ReLU) -> Dropout(0.5) -> Dense(1, sigmoid)` |
| 6. Training, stage 1 | Frozen backbone for rapid convergence | `Adam(lr=1e-4)`, early stopping (patience = 5), `ReduceLROnPlateau` |
| 7. Training, stage 2 | Fine-tune the last 30 convolutional layers | Unfroze layers `-30:`, re-compiled with `Adam(lr=1e-5)` and the same callbacks |
| 8. Evaluation | Tested on the 10% hold-out set | Metrics: accuracy, ROC-AUC, confusion matrix |
| 9. Results | Accuracy ~95.7%, AUC ~0.99 | Confusion matrix: TN = 130, FP = 4, FN = 19, TP = 373 |
| 10. Save | Exported as a single-file archive | `model.save("resnet50_pneumonia.keras")` (~200 MB) |

### Key takeaways

- **High recall on Pneumonia (0.95)** - catches 95% of true cases.
- **Very low false-alarm rate** - only 4 healthy scans flagged as sick.
- Pure Keras `.keras` file, drop-in loadable with `tf.keras.models.load_model`.

## Repository layout

| File | Purpose |
|---|---|
| `pneumonia_detector.ipynb` | End-to-end notebook: data prep, augmentation, two-stage training, evaluation, model export |
| `app.py` | Streamlit app: uploads an X-ray, runs inference, shows the prediction |
| `model_utils.py` | Pure helper functions used by `app.py` (image preprocessing, model loading) - unit tested in `tests/` |
| `tests/` | Pytest suite covering `model_utils.py` |
| `requirements.txt` | Runtime dependencies for the app and notebook |

## About the model file

The trained model (`resnet50_pneumonia.keras`, ~200 MB) is **not committed to this repository** - it's too large for a normal git repo, and too large for Streamlit Cloud's git-based deploy. `model_utils.load_trained_model` checks the project root first; if it's missing there, it downloads the model from Hugging Face Hub instead, provided `HF_MODEL_REPO` is set.

- **Running locally with the file already in the project root:** nothing to configure, this is the original behavior.
- **Running without the file (e.g. a fresh clone, or the deployed app):** set two things before starting:

  ```bash
  export HF_MODEL_REPO="yourusername/pneumonia-detector-resnet50"   # a public HF model repo
  export HF_MODEL_FILENAME="resnet50_pneumonia.keras"                # optional, this is the default
  ```

  The first call downloads and caches it (via `huggingface_hub`, which manages its own local cache), so this only touches the network once per machine, not once per request.

To get a model file at all, **train it yourself**, run `pneumonia_detector.ipynb` end to end (see [Quick start](#quick-start) below), then upload the resulting `.keras` file to a public Hugging Face model repo (huggingface.co → New Model → Files → Add file).

## Quick start

### 1. Train the model

The notebook is written for **Google Colab**, not local Jupyter: the first data cell calls `google.colab.files.upload()`, which only exists in Colab. Running it in local Jupyter will fail at that cell with `ModuleNotFoundError: No module named 'google'` unless you replace that cell with a plain file path.

No dataset or credentials are bundled with this repo, the notebook pulls the [Coronahack Chest X-Ray dataset](https://www.kaggle.com/datasets/praveengovi/coronahack-chest-xraydataset) live from Kaggle. Before running it, you need:

1. A free [Kaggle account](https://www.kaggle.com), with an API token: **Kaggle → your profile → Account → Create New API Token**. This downloads a `kaggle.json` file.
2. Open `pneumonia_detector.ipynb` in Google Colab and run the cells in order:
   - The `files.upload()` cell prompts you to upload that `kaggle.json`, this is what authenticates the next cell's `kaggle datasets download`.
   - The following cells clean, resize, and augment the images, then train the ResNet-50 model in two stages.
3. The last cell exports `resnet50_pneumonia.keras` (~200 MB), downloadable directly or as a zip.

Training end-to-end (dataset download + both fine-tuning stages) takes a while on Colab's free-tier GPU, budget at least 30-60 minutes.

### 2. Run the app

```bash
git clone https://github.com/fadyabirached/pneumonia-detector.git
cd pneumonia-detector
pip install -r requirements.txt

# place resnet50_pneumonia.keras (produced in step 1) in this directory,
# or export HF_MODEL_REPO instead, see "About the model file" above

streamlit run app.py
```

### 3. Deploy your own live demo (Streamlit Community Cloud)

Once the model is uploaded to a public Hugging Face model repo (step 1), this deploys with no code changes:

1. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, click **New app**.
2. Pick this repo, branch `main`, main file path `app.py`.
3. Under **Advanced settings → Secrets**, add:
   ```toml
   HF_MODEL_REPO = "yourusername/pneumonia-detector-resnet50"
   ```
4. Click **Deploy**. First load downloads and caches the model (~200 MB), so budget a minute or two before the app is responsive.

## Running the tests

```bash
pip install pytest
pytest
```

The test suite covers the preprocessing and model-loading logic in `model_utils.py` without requiring the trained model file or the dataset.

## License

See [LICENSE](LICENSE).
