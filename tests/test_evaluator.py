"""Tests for the model-agnostic evaluator."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from adma.evaluator import (
    Detection,
    EvalDataset,
    average_precision,
    box_iou,
    confusion_matrix,
    evaluate,
    image_to_label_path,
    load_dataset,
    load_json_predictions,
    load_yolo_predictions,
    match_detections,
    predict_with,
    save_report,
)

NAMES = {0: "tank", 1: "drone"}
BOX_A = (0.1, 0.1, 0.3, 0.3)
BOX_B = (0.6, 0.6, 0.9, 0.9)


def make_dataset(ground_truth, tmp_path, image_ids=("img1",)):
    images = {i: tmp_path / Path(i).with_suffix(".jpg") for i in image_ids}
    for image_path in images.values():
        write_image(image_path)
    return EvalDataset(NAMES, images, list(ground_truth), tmp_path)


def gt(class_id, bbox, image_id="img1"):
    return Detection(image_id, class_id, bbox)


def pred(class_id, bbox, score, image_id="img1"):
    return Detection(image_id, class_id, bbox, score)


def write_image(path, size=(100, 50)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size).save(path)


def write_yaml(path, cfg):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg))


# ------------------------------------------------------------------
# Metric primitives
# ------------------------------------------------------------------


def test_box_iou_values():
    ious = box_iou([[0, 0, 10, 10]], [[0, 0, 10, 10], [5, 5, 15, 15], [20, 20, 30, 30]])
    np.testing.assert_allclose(ious, [[1.0, 25 / 175, 0.0]])


def test_box_iou_handles_empty_inputs():
    assert box_iou([], [[0, 0, 1, 1]]).shape == (0, 1)


def test_match_detections_assigns_each_gt_once():
    ious = np.array([[0.9], [0.8]])
    tp = match_detections(ious, np.array([0.5]))
    assert tp[:, 0].tolist() == [True, False]


def test_match_detections_per_threshold():
    tp = match_detections(np.array([[0.6]]), np.array([0.5, 0.6, 0.7]))
    assert tp[0].tolist() == [True, True, False]


def test_average_precision_matches_hand_computation():
    # Sorted predictions TP, FP, TP over 2 GT: precision envelope is 1.0 up to
    # recall 0.5 (51 of 101 points) and 2/3 above it (50 points).
    ap = average_precision(np.array([True, False, True]), num_gt=2)
    assert ap == pytest.approx((51 * 1.0 + 50 * 2 / 3) / 101)


def test_average_precision_edge_cases():
    assert average_precision(np.array([], dtype=bool), num_gt=3) == 0.0
    assert average_precision(np.array([False]), num_gt=0) is None


# ------------------------------------------------------------------
# evaluate()
# ------------------------------------------------------------------


def test_perfect_predictions_score_one(tmp_path):
    dataset = make_dataset([gt(0, BOX_A), gt(1, BOX_B)], tmp_path)
    report = evaluate(dataset, [pred(0, BOX_A, 0.9), pred(1, BOX_B, 0.8)])

    assert report.map50_95 == pytest.approx(1.0)
    assert report.map50 == pytest.approx(1.0)
    assert (report.precision, report.recall, report.f1) == (1.0, 1.0, 1.0)


def test_iou_dependent_ap(tmp_path):
    # IoU 0.62 only counts at the 0.50, 0.55 and 0.60 thresholds.
    dataset = make_dataset([gt(0, (0.0, 0.0, 1.0, 1.0))], tmp_path)
    tank = evaluate(dataset, [pred(0, (0.0, 0.0, 0.62, 1.0), 0.9)]).per_class[0]
    assert tank.ap50 == pytest.approx(1.0)
    assert tank.ap75 == 0.0
    assert tank.ap50_95 == pytest.approx(0.3)


def test_duplicate_detection_is_false_positive(tmp_path):
    dataset = make_dataset([gt(0, BOX_A)], tmp_path)
    tank = evaluate(dataset, [pred(0, BOX_A, 0.9), pred(0, BOX_A, 0.8)]).per_class[0]

    assert (tank.tp, tank.fp, tank.fn) == (1, 1, 0)
    assert tank.ap50 == pytest.approx(1.0)


def test_conf_threshold_only_affects_operating_metrics(tmp_path):
    dataset = make_dataset([gt(0, BOX_A), gt(0, BOX_B)], tmp_path)
    preds = [pred(0, BOX_A, 0.9), pred(0, BOX_B, 0.1)]
    tank = evaluate(dataset, preds, conf_threshold=0.5).per_class[0]

    assert tank.ap50 == pytest.approx(1.0)
    assert (tank.num_pred, tank.recall) == (1, 0.5)


def test_class_without_ground_truth_is_excluded_from_means(tmp_path):
    dataset = make_dataset([gt(0, BOX_A)], tmp_path)
    report = evaluate(dataset, [pred(0, BOX_A, 0.9), pred(1, BOX_B, 0.9)])
    drone = report.per_class[1]

    assert drone.ap50_95 is None and drone.fp == 1
    assert report.map50_95 == pytest.approx(1.0)


def test_missed_class_scores_zero(tmp_path):
    dataset = make_dataset([gt(0, BOX_A), gt(1, BOX_B)], tmp_path)
    report = evaluate(dataset, [pred(0, BOX_A, 0.9)])

    assert report.per_class[1].ap50 == 0.0
    assert report.map50 == pytest.approx(0.5)


def test_max_det_keeps_highest_scores(tmp_path):
    dataset = make_dataset([gt(0, BOX_A)], tmp_path)
    preds = [pred(0, BOX_B, 0.5), pred(0, BOX_A, 0.4)]

    assert evaluate(dataset, preds, max_det=1).per_class[0].tp == 0
    assert evaluate(dataset, preds, max_det=2).per_class[0].tp == 1


def test_best_f1_confidence(tmp_path):
    dataset = make_dataset([gt(0, BOX_A)], tmp_path)
    report = evaluate(dataset, [pred(0, BOX_A, 0.8), pred(0, BOX_B, 0.3)])

    assert report.best_f1 == pytest.approx(1.0)
    assert 0.3 < report.best_f1_confidence <= 0.8


def test_unknown_prediction_image_raises(tmp_path):
    dataset = make_dataset([gt(0, BOX_A)], tmp_path)
    with pytest.raises(ValueError, match="not in the dataset"):
        evaluate(dataset, [pred(0, BOX_A, 0.9, image_id="other")])


def test_confusion_matrix_cells(tmp_path):
    dataset = make_dataset([gt(0, BOX_A), gt(1, BOX_B)], tmp_path)
    spurious = (0.4, 0.4, 0.5, 0.5)
    labels, matrix = confusion_matrix(dataset, [pred(1, BOX_A, 0.9), pred(0, spurious, 0.9)])

    assert labels == ["tank", "drone", "background"]
    assert matrix[0, 1] == 1  # tank predicted as drone
    assert matrix[1, 2] == 1  # drone missed
    assert matrix[2, 0] == 1  # spurious tank
    assert matrix.sum() == 3


# ------------------------------------------------------------------
# Loaders
# ------------------------------------------------------------------


@pytest.fixture
def combined_dataset(tmp_path):
    """Combined-style layout with the same filename under two source folders."""
    root = tmp_path / "combined"
    for source in ("tank_images", "military_objects"):
        write_image(root / "images" / "val" / source / "img1.jpg")
    write_image(root / "images" / "val" / "tank_images" / "negative.png")
    labels = root / "labels" / "val"
    (labels / "tank_images").mkdir(parents=True)
    (labels / "military_objects").mkdir(parents=True)
    (labels / "tank_images" / "img1.txt").write_text("0 0.5 0.5 0.2 0.4\n")
    (labels / "military_objects" / "img1.txt").write_text("1 0.1 0.1 0.3 0.1 0.3 0.3\n")
    write_yaml(
        root / "data.yaml",
        {
            "path": str(root),
            "train": ["images/train/tank_images"],
            "val": ["images/val/tank_images", "images/val/military_objects"],
            "names": ["tank", "drone"],
        },
    )
    return root


def test_load_dataset_combined_layout(combined_dataset):
    dataset = load_dataset(combined_dataset / "data.yaml")

    assert set(dataset.images) == {
        "tank_images/img1",
        "tank_images/negative",
        "military_objects/img1",
    }
    by_image = {g.image_id: g for g in dataset.ground_truth}
    assert by_image["tank_images/img1"].bbox == pytest.approx((0.4, 0.3, 0.6, 0.7))
    # Polygon labels are reduced to their bounding box.
    assert by_image["military_objects/img1"].bbox == pytest.approx((0.1, 0.1, 0.3, 0.3))


def test_load_dataset_roboflow_relative_paths(tmp_path):
    root = tmp_path / "tank_images"
    write_image(root / "valid" / "images" / "a.jpg")
    (root / "valid" / "labels").mkdir(parents=True)
    (root / "valid" / "labels" / "a.txt").write_text("0 0.5 0.5 1 1\n")
    write_yaml(root / "data.yaml", {"val": "../valid/images", "names": ["tank"]})

    dataset = load_dataset(root / "data.yaml")
    assert list(dataset.images) == ["a"]
    assert len(dataset.ground_truth) == 1


def test_load_dataset_rejects_out_of_range_class(combined_dataset):
    (combined_dataset / "labels/val/tank_images/img1.txt").write_text("7 0.5 0.5 0.2 0.2\n")
    with pytest.raises(ValueError, match="class id 7"):
        load_dataset(combined_dataset / "data.yaml")


def test_image_to_label_path_only_swaps_images_component():
    path = Path("/data/combined/images/val/tank_images/img1.jpg")
    assert image_to_label_path(path) == Path("/data/combined/labels/val/tank_images/img1.txt")


def test_load_yolo_predictions_nested_and_flat(combined_dataset, tmp_path):
    dataset = load_dataset(combined_dataset / "data.yaml")

    nested = tmp_path / "nested"
    (nested / "tank_images").mkdir(parents=True)
    (nested / "tank_images" / "img1.txt").write_text("0 0.5 0.5 0.2 0.4 0.9\n")
    preds = load_yolo_predictions(nested, dataset)
    assert [(p.image_id, p.score) for p in preds] == [("tank_images/img1", 0.9)]

    flat = tmp_path / "flat"
    flat.mkdir()
    (flat / "negative.txt").write_text("0 0.5 0.5 0.2 0.4 0.7\n")
    assert load_yolo_predictions(flat, dataset)[0].image_id == "tank_images/negative"

    (flat / "img1.txt").write_text("0 0.5 0.5 0.2 0.4 0.7\n")
    with pytest.raises(ValueError, match="ambiguous"):
        load_yolo_predictions(flat, dataset)


def test_load_yolo_predictions_requires_confidence(combined_dataset, tmp_path):
    dataset = load_dataset(combined_dataset / "data.yaml")
    (tmp_path / "negative.txt").write_text("0 0.5 0.5 0.2 0.4\n")
    with pytest.raises(ValueError, match="save_conf"):
        load_yolo_predictions(tmp_path, dataset)


def test_load_json_predictions_normalizes_pixels(combined_dataset, tmp_path):
    dataset = load_dataset(combined_dataset / "data.yaml")
    records = [
        {
            "image": "tank_images/img1.jpg",
            "class": "tank",
            "confidence": 0.9,
            "bbox": [40, 15, 60, 35],
        },
        {"image": "negative.png", "class_id": 1, "score": 0.5, "bbox": [0, 0, 10, 10]},
    ]
    path = tmp_path / "preds.json"
    path.write_text(json.dumps(records))

    preds = load_json_predictions(path, dataset)
    assert preds[0].bbox == pytest.approx((0.4, 0.3, 0.6, 0.7))
    assert preds[1].image_id == "tank_images/negative"


def test_json_class_names_must_match_unless_mapped(combined_dataset, tmp_path):
    dataset = load_dataset(combined_dataset / "data.yaml")
    records = [
        {"image": "negative.png", "class": "Tank", "confidence": 0.9, "bbox": [0, 0, 1, 1]},
        {"image": "negative.png", "class": "person", "confidence": 0.9, "bbox": [0, 0, 1, 1]},
    ]
    path = tmp_path / "preds.json"
    path.write_text(json.dumps(records))

    with pytest.raises(ValueError, match="not a dataset class"):
        load_json_predictions(path, dataset)
    with pytest.warns(UserWarning, match="person"):
        preds = load_json_predictions(path, dataset, class_map={"Tank": "tank"})
    assert [p.class_id for p in preds] == [0]


def test_predict_with_any_callable(combined_dataset):
    dataset = load_dataset(combined_dataset / "data.yaml")
    calls = []

    def predictor(image_path):
        calls.append(image_path)
        return [{"class": "tank", "confidence": 0.9, "bbox": [40, 15, 60, 35]}]

    preds, latency = predict_with(predictor, dataset, warmup=1)
    assert len(calls) == len(dataset.images) + 1
    assert len(preds) == len(dataset.images)
    assert latency.images == len(dataset.images) and latency.fps > 0

    tank = evaluate(dataset, preds, latency=latency).per_class[0]
    assert (tank.tp, tank.fp) == (1, 2)


# ------------------------------------------------------------------
# Reporting and CLI
# ------------------------------------------------------------------


def test_save_report_writes_files(tmp_path):
    dataset = make_dataset([gt(0, BOX_A)], tmp_path)
    report = evaluate(dataset, [pred(0, BOX_A, 0.9)])
    paths = save_report(report, tmp_path / "out", {"model": "test"})

    metrics = json.loads(paths["metrics"].read_text())
    assert metrics["metadata"] == {"model": "test"}
    assert metrics["summary"]["mAP50"] == pytest.approx(1.0)
    assert paths["per_class"].read_text().startswith("class_id,name,num_gt")
    assert "background" in paths["confusion_matrix"].read_text()


def load_cli():
    script = Path(__file__).resolve().parent.parent / "scripts" / "evaluator.py"
    spec = importlib.util.spec_from_file_location("evaluator_cli", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_with_prediction_labels(combined_dataset, tmp_path):
    preds_dir = tmp_path / "preds"
    for source in ("tank_images", "military_objects"):
        (preds_dir / source).mkdir(parents=True)
    (preds_dir / "tank_images" / "img1.txt").write_text("0 0.5 0.5 0.2 0.4 0.9\n")
    (preds_dir / "military_objects" / "img1.txt").write_text("1 0.2 0.2 0.2 0.2 0.8\n")

    out = tmp_path / "report"
    load_cli().main(
        [
            "--data",
            str(combined_dataset / "data.yaml"),
            "--pred-labels",
            str(preds_dir),
            "--output-dir",
            str(out),
        ]
    )

    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["summary"]["mAP50-95"] == pytest.approx(1.0)
    assert metrics["counts"] == {"images": 3, "ground_truth": 2, "predictions": 2}


def test_per_source_reports_perfect_and_missed_sources(tmp_path):
    dataset = make_dataset(
        [
            gt(0, BOX_A, image_id="source_a/img1"),
            gt(0, BOX_A, image_id="source_b/img1"),
        ],
        tmp_path,
        image_ids=("source_a/img1", "source_b/img1"),
    )
    predictions = [
        pred(0, BOX_A, 0.9, image_id="source_a/img1"),
    ]

    report = evaluate(dataset, predictions)

    assert report.per_source["source_a"]["mAP50"] == pytest.approx(1.0)
    assert report.per_source["source_b"]["mAP50"] == pytest.approx(0.0)
    assert report.per_source["source_a"]["recall"] == pytest.approx(1.0)
    assert report.per_source["source_b"]["recall"] == pytest.approx(0.0)


def test_compute_size_metrics_perfect_small_box(tmp_path):
    from adma.evaluator import _compute_size_metrics

    image_path = tmp_path / "small.jpg"
    write_image(image_path, size=(100, 100))

    dataset = EvalDataset(
        class_names=NAMES,
        images={"img1": image_path},
        ground_truth=[gt(0, (0.1, 0.1, 0.2, 0.2))],
        root=tmp_path,
    )
    predictions = [pred(0, (0.1, 0.1, 0.2, 0.2), 0.99)]

    result = _compute_size_metrics(dataset, predictions)

    assert result["small"]["ground_truth"] == 1
    assert result["small"]["AP50"] == 1.0
    assert result["small"]["AP50-95"] == 1.0
    assert result["medium"]["ground_truth"] == 0
    assert result["large"]["ground_truth"] == 0


def test_compute_size_metrics_missed_small_box(tmp_path):
    from adma.evaluator import _compute_size_metrics

    image_path = tmp_path / "small.jpg"
    write_image(image_path, size=(100, 100))

    dataset = EvalDataset(
        class_names=NAMES,
        images={"img1": image_path},
        ground_truth=[gt(0, (0.1, 0.1, 0.2, 0.2))],
        root=tmp_path,
    )

    result = _compute_size_metrics(dataset, [])

    assert result["small"]["ground_truth"] == 1
    assert result["small"]["AP50"] == 0.0
    assert result["small"]["AP50-95"] == 0.0
