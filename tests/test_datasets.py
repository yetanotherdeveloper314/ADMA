"""Core tests for the dataset registry and label normalization."""

from adma.datasets import DATASETS, LABEL_MAP


def test_all_datasets_have_required_keys():
    required = {"workspace", "project", "version", "location", "description"}
    for name, info in DATASETS.items():
        missing = required - set(info.keys())
        assert not missing, f"Dataset '{name}' is missing: {missing}"


def test_cross_dataset_labels_normalize_consistently():
    """Same concepts across datasets must map to the same unified name."""
    assert LABEL_MAP["tank"] == LABEL_MAP["Tank"] == "tank"
    assert LABEL_MAP["drone"] == LABEL_MAP["Drone"] == "drone"
    assert LABEL_MAP["mil_helicopter"] == LABEL_MAP["Helicopter"] == "military_helicopter"


def test_civilian_labels_are_dropped():
    assert LABEL_MAP["person"] is None
    assert LABEL_MAP["Person"] is None
    assert LABEL_MAP["civ_hel"] is None
