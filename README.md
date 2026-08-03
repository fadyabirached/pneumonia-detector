# Pneumonia Detector (ResNet-50)

A Streamlit web app that classifies chest X-ray images as **Normal** or **Pneumonia**, powered by a ResNet-50 model fine-tuned on the [Coronahack Chest X-Ray dataset](https://www.kaggle.com/datasets/praveengovi/coronahack-chest-xraydataset).

![App screenshot](https://github.com/user-attachments/assets/07b6d1be-c822-463a-b140-d25b28e679f2)

## Model pipeline

| Phase | What I did | Details |
|---|---|---|
| 1. Data split | Stratified 80 / 10 / 10 split into train / validation / test | Preserves the Normal / Pneumonia ratio in every subset for fair metrics |
| 2. Pre-processing | 224 x 224 RGB resize + normalization | Keeps input consistent with ImageNet-pretrained networks |
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

The trained model (`resnet50_pneumonia.keras`, ~200 MB) is **not committed to this repository** - it's too large for a normal git repo. `app.py` looks for it in the project root and will show a clear error in the UI if it's missing.

To get a model file, either:

1. **Train it yourself** - run `pneumonia_detector.ipynb` end-to-end (see [Quick start](#quick-start) below), or
2. **Track it with [Git LFS](https://git-lfs.com/)** if you'd rather version large binaries in git - this repo doesn't use LFS today, but it's a reasonable next step if the model needs to live alongside the code.

## Quick start

### 1. Train the model

Open `pneumonia_detector.ipynb` in Jupyter or Google Colab and run all cells. This will:

- download and clean the dataset,
- preprocess and augment the images,
- train the ResNet-50 model in two stages,
- export `resnet50_pneumonia.keras` (downloadable directly or as a zip).

### 2. Run the app

```bash
git clone https://github.com/fadyabirached/pneumonia-detector.git
cd pneumonia-detector
pip install -r requirements.txt

# place resnet50_pneumonia.keras (produced in step 1) in this directory

streamlit run app.py
```

## Running the tests

```bash
pip install pytest
pytest
```

The test suite covers the preprocessing and model-loading logic in `model_utils.py` without requiring the trained model file or the dataset.

## License

See [LICENSE](LICENSE).
