"""Core tests for the detection engine."""

from unittest.mock import patch

import numpy as np
import pytest

from adma.detector import MilitaryAssetDetector, discover_models

SAMPLE_DETECTIONS = [
    {"class": "tank", "confidence": 0.92, "bbox": [10, 20, 100, 120]},
    {"class": "tank", "confidence": 0.85, "bbox": [200, 50, 350, 200]},
    {"class": "drone", "confidence": 0.78, "bbox": [400, 300, 500, 400]},
]


def test_count_by_class():
    counts = MilitaryAssetDetector._count_by_class(SAMPLE_DETECTIONS)

    assert counts["tank"]["count"] == 2
    assert counts["drone"]["count"] == 1
    assert counts["tank"]["confidences"] == [0.92, 0.85]


def test_build_summary_with_detections():
    counts = MilitaryAssetDetector._count_by_class(SAMPLE_DETECTIONS)
    summary = MilitaryAssetDetector._build_summary(SAMPLE_DETECTIONS, counts)

    assert "Detected 3 military asset(s)" in summary
    assert "tank" in summary
    assert "drone" in summary


def test_build_summary_empty():
    assert "No military assets" in MilitaryAssetDetector._build_summary([], {})


def test_draw_boxes_does_not_mutate_original():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    original = img.copy()
    detector = object.__new__(MilitaryAssetDetector)
    result = detector._draw_boxes(img, SAMPLE_DETECTIONS)

    np.testing.assert_array_equal(img, original)
    assert not np.array_equal(result, original)


def test_discover_models_finds_best_pt(tmp_path):
    (tmp_path / "my_model").mkdir()
    (tmp_path / "my_model" / "best.pt").touch()

    with patch("adma.detector.MODELS_DIR", str(tmp_path)):
        models = discover_models()

    assert "my_model" in models


def test_detector_raises_on_missing_model():
    with pytest.raises(FileNotFoundError):
        MilitaryAssetDetector("/nonexistent/best.pt")
