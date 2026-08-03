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


def test_prepare_image_produces_expected_shape_and_dtype(monkeypatch):
    # Stand in for tensorflow.keras.applications.resnet50.preprocess_input
    # so this test doesn't need a real TensorFlow install.
    fake_resnet50 = types.ModuleType("tensorflow.keras.applications.resnet50")
    fake_resnet50.preprocess_input = lambda x: x  # identity, we only check shape/dtype
    monkeypatch.setitem(sys.modules, "tensorflow.keras.applications.resnet50", fake_resnet50)

    # Non-square grayscale input, like a raw chest X-ray, to exercise the
    # RGB conversion + resize path.
    img = Image.new("L", (300, 150), color=128)

    batch = model_utils.prepare_image(img)

    assert batch.shape == (1, 224, 224, 3)
    assert batch.dtype == np.float32


def test_prepare_image_calls_resnet_preprocess_input(monkeypatch):
    calls = []

    def fake_preprocess_input(x):
        calls.append(x.shape)
        return x

    fake_resnet50 = types.ModuleType("tensorflow.keras.applications.resnet50")
    fake_resnet50.preprocess_input = fake_preprocess_input
    monkeypatch.setitem(sys.modules, "tensorflow.keras.applications.resnet50", fake_resnet50)

    img = Image.new("RGB", (50, 50), color=(10, 20, 30))
    model_utils.prepare_image(img)

    assert calls == [(1, 224, 224, 3)]


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
    # app.py treats index 1 / the positive class as "Pneumonia" — pin this
    # down since a reordering would silently flip predictions.
    assert model_utils.CLASS_NAMES == ["Normal", "Pneumonia"]
