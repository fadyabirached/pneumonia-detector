![Screenshot 2025-05-30 105032](https://github.com/user-attachments/assets/07b6d1be-c822-463a-b140-d25b28e679f2)
# 🩺 Pneumonia-Detector (ResNet-50)

A Streamlit web-app that classifies chest X-ray images as **Normal** or **Pneumonia** using a fine-tuned ResNet-50 model.

---

## Quick start

```bash
git clone https://github.com/<your-user>/pneumonia-detector.git
cd pneumonia-detector
# drop the model file next to app.py
streamlit run app.py
```
## 📈 Model-building pipeline

| Phase | What I did | 🔍 Details |
|-------|-------------|-----------|
| **1. Data split** | Stratified 80 / 10 / 10 split → **train**, **validation**, **test** | Preserves Normal / Pneumonia ratio in every subset for fair metrics |
| **2. Pre-processing** | *224 × 224 RGB* resize + normalization | Keeps input consistent with ImageNet-pretrained networks |
| **3. Data augmentation** | `ImageDataGenerator` on the fly | `brightness_range=[0.7, 1.3]`, `zoom_range=0.2`, `horizontal_flip=True` |
| **4. Class weights** | Inverse-frequency `{0: w_normal, 1: w_pneu}` | Penalises misclassifying the rarer class (Normal) |
| **5. Architecture** | ResNet-50 **backbone** (ImageNet weights, frozen) + custom head | `GlobalAveragePooling2D → BatchNorm → Dense(256, ReLU) → Dropout(0.5) → Dense(1, sigmoid)` |
| **6. Training – stage 1** | **Frozen** backbone for rapid convergence | `Adam(lr = 1e-4)`,  early stopping (patience = 5), `ReduceLROnPlateau` |
| **7. Training – stage 2** | **Fine-tune** last 30 convolutional layers | Unfroze layers -30: , re-compiled with `Adam(lr = 1e-5)` and same callbacks |
| **8. Evaluation** | Tested on the 10 % hold-out set | Metrics: Accuracy, ROC-AUC, confusion matrix |
| **9. Results** | Accuracy ≈ **95.7 %**, AUC ≈ **0.99** | Confusion matrix:<br> TN = 130   FP = 4  <br> FN = 19   TP = 373 |
| **10. Save** | Exported single-file archive | `model.save("resnet50_pneumonia.keras")` (≈200 MB) |

### Key take-aways
* **High recall on Pneumonia (0.95)** – catches 95 % of true cases.  
* **Very low false-alarm rate** – only 4 healthy scans flagged as sick.  
* Pure Keras `.keras` file → drop-in loadable with `tf.keras.models.load_model`.

```bash
# quick load example
from tensorflow.keras.models import load_model
model = load_model("resnet50_pneumonia.keras")
