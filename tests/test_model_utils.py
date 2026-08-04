"""Unit tests for model_utils.py.

These run without the trained model file, the X-ray dataset, or a real
TensorFlow install: the TF calls used inside `prepare_image` and
`load_trained_model` are stubbed out via `sys.modules`.
"""
import sys
import types

import numpy as np
import pytest
from PIL import Image

import model_utils


def test_predicted_class_below_threshold_is_normal():
    assert model_utils.predicted_class(0.2, threshold=0.5) == "Normal"


def test_predicted_class_above_threshold_is_pneumonia():
    assert model_utils.predicted_class(0.8, threshold=0.5) == "Pneumonia"


def test_predicted_class_boundary_is_inclusive_of_positive_class():
    assert model_utils.predicted_class(0.5, threshold=0.5) == "Pneumonia"


def test_predicted_class_respects_custom_threshold():
    assert model_utils.predicted_class(0.4, threshold=0.3) == "Pneumonia"
    assert model_utils.predicted_class(0.4, threshold=0.6) == "Normal"


def test_prepare_image_produces_expected_shape_and_dtype():
    # Non-square grayscale input, like a raw chest X-ray, to exercise the
    # RGB conversion + resize path.
    img = Image.new("L", (300, 150), color=128)

    batch = model_utils.prepare_image(img)

    assert batch.shape == (1, 224, 224, 3)
    assert batch.dtype == np.float32


def test_prepare_image_leaves_pixel_values_unrescaled():
    # Regression test: the training notebook's ImageDataGenerator calls
    # (cell 10) set neither `rescale` nor `preprocessing_function`, so the
    # model trained on raw 0-255 pixel values. prepare_image used to run
    # inputs through keras.applications.resnet50.preprocess_input, which
    # RGB-to-BGR-converts and ImageNet-mean-subtracts, a distribution the
    # model never saw. A flat mid-grey input should come back as exactly
    # that value, not shifted/reordered by a preprocessing step the
    # training pipeline never applied.
    img = Image.new("RGB", (50, 50), color=(200, 100, 50))

    batch = model_utils.prepare_image(img)

    assert batch[0, 0, 0, 0] == pytest.approx(200.0)  # R
    assert batch[0, 0, 0, 1] == pytest.approx(100.0)  # G
    assert batch[0, 0, 0, 2] == pytest.approx(50.0)   # B
    assert batch.min() >= 0.0
    assert batch.max() <= 255.0


def test_prepare_image_needs_no_tensorflow_import(monkeypatch):
    # prepare_image previously imported keras.applications.resnet50 for
    # preprocess_input. Blocking `import tensorflow` entirely and confirming
    # this still works proves that import is gone for good, not just
    # unused by coincidence in this test run.
    monkeypatch.setitem(sys.modules, "tensorflow", None)

    img = Image.new("RGB", (10, 10), color=(1, 2, 3))
    batch = model_utils.prepare_image(img)

    assert batch.shape == (1, 224, 224, 3)


def test_load_trained_model_missing_file_raises_clear_error(tmp_path):
    missing_path = tmp_path / "does_not_exist.keras"

    with pytest.raises(FileNotFoundError, match="Model file not found"):
        model_utils.load_trained_model(str(missing_path))


def test_load_trained_model_delegates_to_keras_load_model(monkeypatch, tmp_path):
    model_file = tmp_path / "fake_model.keras"
    model_file.write_text("stand-in for a real .keras archive")

    sentinel_model = object()
    fake_models_module = types.ModuleType("tensorflow.keras.models")
    fake_models_module.load_model = lambda path: sentinel_model
    monkeypatch.setitem(sys.modules, "tensorflow.keras.models", fake_models_module)

    result = model_utils.load_trained_model(str(model_file))

    assert result is sentinel_model


def test_class_names_order_matches_pneumonia_positive_class():
    # app.py treats index 1 / the positive class as "Pneumonia", pin this
    # down since a reordering would silently flip predictions.
    assert model_utils.CLASS_NAMES == ["Normal", "Pneumonia"]
