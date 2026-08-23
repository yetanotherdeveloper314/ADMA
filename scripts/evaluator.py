"""
Model-Agnostic Object Detection Evaluator
==========================================

Computes COCO AP, AP50, AP75, Precision, Recall, F1, and a confusion matrix from any
set of predictions and ground truth boxes, regardless of which model
produced the predictions. This module has NO dependency on train.py, any
training framework, -- it only needs
YOLO-format label files (and, for the standalone CLI, a data.yaml for
class names). It can be imported as a library or run directly:

    python evaluator.py \
        --data data.yaml \
        --gt-images path/to/val/images --gt-labels path/to/val/labels \
        --pred-labels path/to/predicted/labels \
        --output-dir path/to/output

INPUT FORMAT (internal, model-agnostic)
----------------------------------------
Ground truth : List[dict] with keys:
    image_id : str   (unique per image -- see compute_image_id)
    class_id : int
    bbox     : [x1, y1, x2, y2]   (absolute pixel coords)

Predictions  : List[dict] with keys:
    image_id : str
    class_id : int
    bbox     : [x1, y1, x2, y2]
    score    : float              (confidence, 0-1)

Loader provided for YOLO-format datasets (load_yolo). For a different
detector's raw output (e.g. a live model's .detect() call), write a small
adapter -- see predictions_from_detector -- that maps it into the same
list-of-dicts shape; everything else works unchanged.

IMAGE IDENTIFIERS
------------------
Every image's id is its path relative to a shared dataset_root, with the
extension stripped (see compute_image_id). This matters once you have more
than one images/ subfolder feeding into the same evaluation (e.g. a
combined dataset with per-source subfolders like "tank_images",
"military_vehicles", ...): a bare filename stem can collide across
subfolders ("img001.jpg" existing under two different sources), silently
merging two different images' ground truth under one id. Passing a shared
dataset_root when loading each subfolder keeps ids unique.

USAGE (library)
-----
    from evaluator import (
        DetectionEvaluator, load_yolo, load_class_names, load_class_name_to_id, load_dataset_split,
    )

    class_names = load_class_names("data.yaml")
    ground_truth = load_dataset_split("data.yaml", split="val")
    predictions = load_yolo("runs/predict/labels", "images/val",
                             dataset_root="images/val", is_prediction=True)

    evaluator = DetectionEvaluator(ground_truth, predictions, class_names)
    report = evaluator.evaluate()
    evaluator.print_report(report)
    cm = evaluator.confusion_matrix()
    evaluator.print_confusion_matrix(cm)
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from ultralytics import YOLO


# --------------------------------------------------------------------------
# IoU utility (pure)
# --------------------------------------------------------------------------
def compute_iou(box_a, box_b) -> float:
    """IoU between two [x1,y1,x2,y2] boxes. Pure function."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
    ix2, iy2 = min(xa2, xb2), min(ya2, yb2)

    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compute_ious(box, boxes) -> np.ndarray:
    """Vectorized IoU between one box and many [x1,y1,x2,y2] boxes."""
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.size == 0:
        return np.empty(0, dtype=np.float32)

    boxes = boxes.reshape(-1, 4)
    box = np.asarray(box, dtype=np.float32)

    ix1 = np.maximum(box[0], boxes[:, 0])
    iy1 = np.maximum(box[1], boxes[:, 1])
    ix2 = np.minimum(box[2], boxes[:, 2])
    iy2 = np.minimum(box[3], boxes[:, 3])

    iw = np.maximum(0.0, ix2 - ix1)
    ih = np.maximum(0.0, iy2 - iy1)
    inter = iw * ih

    area_box = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    area_boxes = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0, boxes[:, 3] - boxes[:, 1]
    )
    union = area_box + area_boxes - inter

    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


# --------------------------------------------------------------------------
# Image identifiers (pure)
# --------------------------------------------------------------------------
def compute_image_id(image_path, dataset_root) -> str:
    """
    Unique identifier for an image: its path relative to dataset_root,
    extension stripped, normalized to forward slashes. Pure function
    given two path-like inputs (does not require the paths to exist).

    Using a path relative to a shared root -- rather than a bare filename
    stem -- avoids collisions when the same filename appears under
    different sibling subfolders (a real risk in a combined dataset
    built from several source datasets).

    Falls back to just the filename (stem) if image_path is not actually
    under dataset_root, so callers get a stable id instead of a hard
    failure.
    """
    image_path = Path(image_path).resolve()
    dataset_root = Path(dataset_root).resolve()
    try:
        rel = image_path.relative_to(dataset_root)
    except ValueError:
        rel = Path(image_path.name)
    return rel.with_suffix("").as_posix()


def images_dir_to_labels_dir(images_dir) -> Path:
    """
    Mirror an images path into its labels path by swapping only the
    exact 'images' path COMPONENT, not any substring match. Pure
    function. A naive str.replace("images", "labels") mangles folder
    names like 'tank_images' into 'tank_labels' -- this only touches
    path segments that are exactly 'images'.
    """
    images_dir = Path(images_dir)
    swapped = ["labels" if p == "images" else p for p in images_dir.parts]
    return Path(*swapped)


# --------------------------------------------------------------------------
# data.yaml helpers
# --------------------------------------------------------------------------
def load_class_names(data_yaml_path) -> dict[int, str]:
    """
    Load {class_id: name} from a YOLO data.yaml file, e.g.:

        names:
          - armored_vehicle
          - drone
          - tank

    Also supports the dict form:

        names:
          0: armored_vehicle
          1: drone
    """
    with open(data_yaml_path) as f:
        data = yaml.safe_load(f)

    names = data["names"]
    if isinstance(names, list):
        return {i: n for i, n in enumerate(names)}
    return {int(k): v for k, v in names.items()}


def load_class_name_to_id(data_yaml_path) -> dict[str, int]:
    """
    Load {class_name: class_id} from the same YOLO data.yaml used by
    the ground truth, so detector class names map to the exact dataset IDs.
    """
    class_names = load_class_names(data_yaml_path)

    class_name_to_id: dict[str, int] = {}
    for class_id, name in class_names.items():
        if name in class_name_to_id:
            raise ValueError(
                f"Duplicate class name {name!r} in {data_yaml_path}; "
                "cannot build an unambiguous class-name mapping."
            )
        class_name_to_id[name] = class_id

    return class_name_to_id


def resolve_split_dirs(data_yaml_path, split: str = "val") -> list[tuple[Path, Path]]:
    """
    Return (images_dir, labels_dir) pairs for a split (default "val")
    described by a data.yaml file. Handles a single-dataset yaml
    (val: a string) and a combined-dataset yaml (val: a list of paths,
    one per source dataset, e.g. images/val/tank_images,
    images/val/military_vehicles, ...).
    """
    data_yaml_path = Path(data_yaml_path)
    with open(data_yaml_path) as f:
        cfg = yaml.safe_load(f)

    base = Path(cfg.get("path", str(data_yaml_path.parent.resolve())))
    entries = cfg[split] if isinstance(cfg[split], list) else [cfg[split]]

    pairs = []
    for entry in entries:
        images_dir = base / entry
        labels_dir = images_dir_to_labels_dir(images_dir)
        pairs.append((images_dir, labels_dir))
    return pairs


def load_dataset_split(
    data_yaml_path,
    split: str = "val",
    dataset_root=None,
    is_prediction: bool = False,
) -> list[dict]:
    """
    Convenience loader: resolve every images/labels directory pair for a
    split in a data.yaml (single or combined form) and load YOLO-format
    annotations from all of them into one flat list, sharing a single
    dataset_root so image ids stay unique and comparable across sibling
    source-dataset subfolders. Missing directories are skipped silently
    (mirrors partially-downloaded datasets).
    """
    data_yaml_path = Path(data_yaml_path)
    if dataset_root is None:
        with open(data_yaml_path) as f:
            cfg = yaml.safe_load(f)
        dataset_root = cfg.get("path", str(data_yaml_path.parent.resolve()))
    dataset_root = Path(dataset_root)

    out: list[dict] = []
    for images_dir, labels_dir in resolve_split_dirs(data_yaml_path, split):
        if not images_dir.exists() or not labels_dir.exists():
            continue
        out.extend(
            load_yolo(
                str(labels_dir),
                str(images_dir),
                dataset_root=dataset_root,
                is_prediction=is_prediction,
            )
        )
    return out


# --------------------------------------------------------------------------
# YOLO loader (model-agnostic front door for ground truth / predictions)
# --------------------------------------------------------------------------
def load_yolo(labels_dir, images_dir, dataset_root=None, is_prediction: bool = False) -> list[dict]:
    """
    Load ground truth or predictions from YOLO-style txt files.

    Optimizations:
      * Builds an image-stem index once instead of probing every extension
        for every label file.
      * Reads only image metadata with Pillow to obtain width/height instead
        of fully decoding every image with OpenCV.
    """
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    root = Path(dataset_root) if dataset_root is not None else images_dir

    # Average O(1) image lookup by stem after one directory scan.
    image_index = {
        path.stem: path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in img_exts
    }

    out: list[dict] = []
    for label_path in labels_dir.glob("*.txt"):
        img_path = image_index.get(label_path.stem)
        if img_path is None:
            continue

        try:
            with Image.open(img_path) as image:
                W, H = image.size
        except (OSError, ValueError):
            continue

        image_id = compute_image_id(img_path, root)

        with label_path.open() as f:
            for line in f:
                parts = line.split()
                if len(parts) < 5:
                    continue

                try:
                    cls = int(float(parts[0]))
                    xc, yc, w, h = map(float, parts[1:5])
                except ValueError:
                    continue

                x1 = (xc - w / 2) * W
                y1 = (yc - h / 2) * H
                x2 = (xc + w / 2) * W
                y2 = (yc + h / 2) * H

                entry = {
                    "image_id": image_id,
                    "class_id": cls,
                    "bbox": [x1, y1, x2, y2],
                }

                if is_prediction:
                    try:
                        entry["score"] = float(parts[5]) if len(parts) > 5 else 1.0
                    except ValueError:
                        entry["score"] = 1.0

                out.append(entry)

    return out


# --------------------------------------------------------------------------
# Prediction adapter for detector-style output, e.g.
# {"class": "tank", "confidence": 0.94, "bbox": [x1,y1,x2,y2]}
# --------------------------------------------------------------------------
def predictions_from_detector(
    results,
    class_name_to_id: dict[str, int],
) -> list[dict]:
    """
    Convert detector-style output into the evaluator's internal format.

    IMPORTANT:
    class_name_to_id must come from the SAME data.yaml used for ground truth:

        class_name_to_id = load_class_name_to_id("data.yaml")
        predictions = predictions_from_detector(results, class_name_to_id)

    The evaluator intentionally does not assign IDs in first-seen order,
    because that can silently make prediction IDs disagree with data.yaml.
    """
    if not class_name_to_id:
        raise ValueError(
            "class_name_to_id is required. Build it with load_class_name_to_id(data_yaml_path)."
        )

    def _class_id(name):
        if name not in class_name_to_id:
            raise ValueError(
                f"Unknown detector class {name!r}. Expected one of: {sorted(class_name_to_id)}"
            )
        return class_name_to_id[name]

    out: list[dict] = []

    if isinstance(results, dict):
        for image_id, dets in results.items():
            for det in dets:
                out.append(
                    {
                        "image_id": image_id,
                        "class_id": _class_id(det["class"]),
                        "bbox": list(det["bbox"]),
                        "score": float(det["confidence"]),
                    }
                )
    else:
        for det in results:
            out.append(
                {
                    "image_id": det["image_id"],
                    "class_id": _class_id(det["class"]),
                    "bbox": list(det["bbox"]),
                    "score": float(det["confidence"]),
                }
            )

    return out


# --------------------------------------------------------------------------
# Pure computational core (extracted so it's directly unit-testable
# without constructing a DetectionEvaluator or any real boxes/images)
# --------------------------------------------------------------------------
def compute_average_precision(tp: np.ndarray, fp: np.ndarray, n_gt: int):
    """
    COCO-style Average Precision for one class at one IoU threshold.

    Predictions must already be sorted by descending confidence.
    Precision is interpolated on the standard 101 recall thresholds
    [0.00, 0.01, ..., 1.00], matching the core COCO AP convention.

    Returns:
        (ap, final_precision, final_recall)

    AP is None when n_gt == 0 because the metric is undefined for a class
    with no ground-truth instances.
    """
    if n_gt == 0:
        return None, 0.0, 0.0

    tp_cum = np.cumsum(tp, dtype=np.float64)
    fp_cum = np.cumsum(fp, dtype=np.float64)

    recalls = tp_cum / n_gt
    precisions = tp_cum / np.maximum(
        tp_cum + fp_cum,
        np.finfo(np.float64).eps,
    )

    # COCO precision envelope: precision at recall r is the maximum
    # precision observed at any recall >= r.
    if precisions.size:
        precisions = np.maximum.accumulate(precisions[::-1])[::-1]

    # COCO uses 101 recall thresholds: 0.00, 0.01, ..., 1.00.
    recall_thresholds = np.linspace(0.0, 1.0, 101)
    precision_at_recall = np.zeros(101, dtype=np.float64)

    if recalls.size:
        indices = np.searchsorted(recalls, recall_thresholds, side="left")
        valid = indices < precisions.size
        precision_at_recall[valid] = precisions[indices[valid]]

    ap = float(precision_at_recall.mean())

    final_p = (
        float(
            tp_cum[-1]
            / max(
                tp_cum[-1] + fp_cum[-1],
                np.finfo(np.float64).eps,
            )
        )
        if tp_cum.size
        else 0.0
    )
    final_r = float(tp_cum[-1] / n_gt) if tp_cum.size else 0.0
    return ap, final_p, final_r


def compute_precision_recall_f1(tp: float, fp: float, fn: float):
    """Pure function: precision/recall/F1 from TP/FP/FN counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def match_predictions_to_ground_truth(gt_boxes, pred_boxes, iou_thresh: float = 0.5):
    """
    Pure function. Greedily match predictions (already sorted by
    descending confidence) to ground-truth boxes for ONE image, across
    ALL classes at once, purely by spatial IoU (class-agnostic) -- used
    to build a confusion matrix, where a spatial match with a mismatched
    class should show up as a misclassification, not simply as a miss.

    gt_boxes   : list of (class_id, bbox)
    pred_boxes : list of (class_id, bbox, score), assumed already sorted
                 by score descending
    iou_thresh : minimum IoU to count as a spatial match

    Returns a list of (gt_class_id, pred_class_id) pairs where either
    side may be None:
        (gt_class, pred_class) -- matched detection (correct class if
                                   equal, else a misclassification)
        (gt_class, None)       -- missed ground truth (false negative)
        (None, pred_class)     -- spurious prediction (false positive)
    """
    matched_gt = [False] * len(gt_boxes)
    pairs = []

    for pred_class, pred_box, _score in pred_boxes:
        best_iou, best_j = 0.0, -1
        for j, (_gt_class, gt_box) in enumerate(gt_boxes):
            if matched_gt[j]:
                continue
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou, best_j = iou, j

        if best_iou >= iou_thresh and best_j >= 0:
            matched_gt[best_j] = True
            pairs.append((gt_boxes[best_j][0], pred_class))
        else:
            pairs.append((None, pred_class))

    for j, was_matched in enumerate(matched_gt):
        if not was_matched:
            pairs.append((gt_boxes[j][0], None))

    return pairs


# --------------------------------------------------------------------------
# Core evaluator
# --------------------------------------------------------------------------
class DetectionEvaluator:
    def __init__(self, ground_truths, predictions, class_names: dict[int, str] | None = None):
        """
        ground_truths, predictions : lists of dicts (see module docstring)
        class_names : optional {class_id: name} for a readable report
                      (e.g. from load_class_names("data.yaml"))

        Performance optimizations:
          * Ground truth and predictions are indexed once by class/image.
          * Predictions are sorted by confidence once per class.
          * Match results are cached by (class_id, IoU threshold).
          * Confusion-matrix image groupings are built once.
        """
        self.gt = ground_truths
        self.class_names = class_names or {}
        self.max_detections_per_image = 100  # COCO maxDets=100 benchmark setting

        # COCO evaluates at most the top 100 detections per image, ranked by confidence.
        preds_grouped = defaultdict(list)
        for p in predictions:
            preds_grouped[p["image_id"]].append(p)

        retained_predictions = []
        for image_preds in preds_grouped.values():
            image_preds.sort(key=lambda p: p["score"], reverse=True)
            retained_predictions.extend(image_preds[: self.max_detections_per_image])

        self.preds = retained_predictions
        self.class_names = class_names or {}
        self.classes = sorted(
            {g["class_id"] for g in ground_truths} | {p["class_id"] for p in retained_predictions}
        )
        self.iou_thresholds = np.round(np.arange(0.50, 1.00, 0.05), 2)

        self.gt_by_class_image = defaultdict(lambda: defaultdict(list))
        self.pred_by_class = defaultdict(list)
        self.gt_by_image = defaultdict(list)
        self.pred_by_image = defaultdict(list)

        for g in ground_truths:
            class_id = g["class_id"]
            image_id = g["image_id"]
            bbox = g["bbox"]
            self.gt_by_class_image[class_id][image_id].append(bbox)
            self.gt_by_image[image_id].append((class_id, bbox))

        for p in retained_predictions:
            class_id = p["class_id"]
            image_id = p["image_id"]
            bbox = p["bbox"]
            score = p["score"]
            self.pred_by_class[class_id].append(p)
            self.pred_by_image[image_id].append((class_id, bbox, score))

        for class_preds in self.pred_by_class.values():
            class_preds.sort(key=lambda p: p["score"], reverse=True)

        for image_preds in self.pred_by_image.values():
            image_preds.sort(key=lambda x: x[2], reverse=True)

        self._match_cache: dict[tuple[int, float], tuple[np.ndarray, np.ndarray, int]] = {}

    # ---- matching at a single IoU threshold, single class -------------
    def _match_single_class(self, class_id, iou_thresh):
        """
        Greedily match predictions for one class at one IoU threshold.

        Compared with the original implementation, this method does not
        rescan all GT/predictions or re-sort predictions on every call.
        It also computes all candidate IoUs for a prediction in one
        vectorized NumPy operation.
        """
        key = (int(class_id), float(iou_thresh))
        cached = self._match_cache.get(key)
        if cached is not None:
            return cached

        gt_by_image = self.gt_by_class_image.get(class_id, {})
        class_preds = self.pred_by_class.get(class_id, [])

        matched = {
            image_id: np.zeros(len(boxes), dtype=bool) for image_id, boxes in gt_by_image.items()
        }
        n_gt = sum(len(boxes) for boxes in gt_by_image.values())

        tp = np.zeros(len(class_preds), dtype=np.float32)
        fp = np.zeros(len(class_preds), dtype=np.float32)

        for i, pred in enumerate(class_preds):
            image_id = pred["image_id"]
            gt_boxes = gt_by_image.get(image_id, [])

            if not gt_boxes:
                fp[i] = 1.0
                continue

            ious = compute_ious(pred["bbox"], gt_boxes)
            already_matched = matched[image_id]

            # Exclude GT boxes already assigned to a higher-confidence prediction.
            available_ious = np.where(already_matched, -1.0, ious)
            best_j = int(np.argmax(available_ious))
            best_iou = float(available_ious[best_j])

            if best_iou >= iou_thresh:
                tp[i] = 1.0
                already_matched[best_j] = True
            else:
                fp[i] = 1.0

        result = (tp, fp, n_gt)
        self._match_cache[key] = result
        return result

    def _average_precision(self, class_id, iou_thresh):
        tp, fp, n_gt = self._match_single_class(class_id, iou_thresh)
        return compute_average_precision(tp, fp, n_gt)

    def _prf_at_50(self, class_id):
        tp, fp, n_gt = self._match_single_class(class_id, 0.50)
        tp_sum, fp_sum = float(tp.sum()), float(fp.sum())
        fn_sum = n_gt - tp_sum
        return compute_precision_recall_f1(tp_sum, fp_sum, fn_sum)

    # ---- full report -----------------------------------------------------
    def evaluate(self):
        """Compute COCO-style AP metrics plus Precision/Recall/F1 at IoU=0.50."""
        per_class = {}

        for c in self.classes:
            # Reuse the IoU=0.50 match result for AP50 and P/R/F1.
            tp50, fp50, n_gt = self._match_single_class(c, 0.50)
            ap50, _, _ = compute_average_precision(tp50, fp50, n_gt)

            tp_sum = float(tp50.sum())
            fp_sum = float(fp50.sum())
            fn_sum = n_gt - tp_sum
            precision, recall, f1 = compute_precision_recall_f1(tp_sum, fp_sum, fn_sum)

            ap_per_iou = []
            ap75 = None
            for thr in self.iou_thresholds:
                thr = float(thr)
                if thr == 0.50:
                    ap = ap50
                else:
                    tp, fp, n_gt_thr = self._match_single_class(c, thr)
                    ap, _, _ = compute_average_precision(tp, fp, n_gt_thr)

                if thr == 0.75:
                    ap75 = ap
                if ap is not None:
                    ap_per_iou.append(ap)

            coco_ap = float(np.mean(ap_per_iou)) if ap_per_iou else 0.0

            per_class[c] = {
                "name": self.class_names.get(c, str(c)),
                "AP": coco_ap,
                "AP50": ap50 if ap50 is not None else 0.0,
                "AP75": ap75 if ap75 is not None else 0.0,
                # Backward-compatible alias for existing consumers.
                "mAP50-95": coco_ap,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "n_gt": n_gt,
            }

        # COCO AP is undefined for classes with no ground truth, so exclude them.
        valid = [v for v in per_class.values() if v["n_gt"] > 0]
        overall_ap = float(np.mean([v["AP"] for v in valid])) if valid else 0.0
        overall = {
            "AP": overall_ap,
            "AP50": float(np.mean([v["AP50"] for v in valid])) if valid else 0.0,
            "AP75": float(np.mean([v["AP75"] for v in valid])) if valid else 0.0,
            # Backward-compatible aliases.
            "mAP50-95": overall_ap,
            "mAP50": float(np.mean([v["AP50"] for v in valid])) if valid else 0.0,
            "Precision": float(np.mean([v["Precision"] for v in valid])) if valid else 0.0,
            "Recall": float(np.mean([v["Recall"] for v in valid])) if valid else 0.0,
            "F1": float(np.mean([v["F1"] for v in valid])) if valid else 0.0,
        }
        return {"per_class": per_class, "overall": overall}

    # ---- confusion matrix -------------------------------------------------
    def confusion_matrix(self, iou_thresh: float = 0.5) -> dict:
        """
        Build a (n_classes+1) x (n_classes+1) confusion matrix at a fixed
        IoU threshold. Rows = ground-truth class, columns = predicted class.
        The final row/column is background for unmatched predictions/GT.
        """
        labels = [self.class_names.get(c, str(c)) for c in self.classes] + ["background"]
        idx = {c: i for i, c in enumerate(self.classes)}
        bg = len(self.classes)
        matrix = np.zeros((len(labels), len(labels)), dtype=int)

        image_ids = set(self.gt_by_image) | set(self.pred_by_image)
        for image_id in image_ids:
            gts = self.gt_by_image.get(image_id, [])
            preds = self.pred_by_image.get(image_id, [])
            pairs = match_predictions_to_ground_truth(gts, preds, iou_thresh)

            for gt_class, pred_class in pairs:
                row = idx[gt_class] if gt_class is not None else bg
                col = idx[pred_class] if pred_class is not None else bg
                matrix[row, col] += 1

        return {"labels": labels, "matrix": matrix}

    @staticmethod
    def print_confusion_matrix(cm: dict) -> None:
        labels = cm["labels"]
        matrix = cm["matrix"]
        col_w = max(10, max(len(label) for label in labels) + 2)

        header = " " * col_w + "".join(f"{label[: col_w - 1]:>{col_w}}" for label in labels)
        print("\nConfusion matrix (rows = actual, cols = predicted)")
        print(header)
        for i, row_label in enumerate(labels):
            row_str = f"{row_label[: col_w - 1]:<{col_w}}" + "".join(
                f"{v:>{col_w}}" for v in matrix[i]
            )
            print(row_str)

    @staticmethod
    def save_confusion_matrix_csv(cm: dict, path) -> None:
        labels = cm["labels"]
        matrix = cm["matrix"]
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([""] + labels)
            for i, row_label in enumerate(labels):
                writer.writerow([row_label] + list(matrix[i]))

    @staticmethod
    def print_report(report):
        print(
            f"{'Class':<20}{'COCO AP':>10}{'AP50':>8}{'AP75':>8}"
            f"{'Precision':>12}{'Recall':>10}{'F1':>8}"
        )
        print("-" * 76)
        for _c, v in report["per_class"].items():
            print(
                f"{v['name']:<20}{v['AP']:>10.3f}{v['AP50']:>8.3f}{v['AP75']:>8.3f}"
                f"{v['Precision']:>12.3f}{v['Recall']:>10.3f}{v['F1']:>8.3f}"
            )
        print("-" * 76)
        o = report["overall"]
        print(
            f"{'ALL (COCO mean)':<20}{o['AP']:>10.3f}{o['AP50']:>8.3f}{o['AP75']:>8.3f}"
            f"{o['Precision']:>12.3f}{o['Recall']:>10.3f}{o['F1']:>8.3f}"
        )

    @staticmethod
    def save_json(report, path):
        """Write the full report (per-class + overall) to a JSON file."""
        serializable = {
            "per_class": {str(k): v for k, v in report["per_class"].items()},
            "overall": report["overall"],
        }
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)

    @staticmethod
    def save_csv(report, path):
        """Write COCO-style per-class metrics plus an ALL row to CSV."""
        fieldnames = [
            "class_id",
            "name",
            "AP",
            "AP50",
            "AP75",
            "Precision",
            "Recall",
            "F1",
        ]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for class_id, v in report["per_class"].items():
                writer.writerow(
                    {
                        "class_id": class_id,
                        "name": v["name"],
                        "AP": v["AP"],
                        "AP50": v["AP50"],
                        "AP75": v["AP75"],
                        "Precision": v["Precision"],
                        "Recall": v["Recall"],
                        "F1": v["F1"],
                    }
                )
            o = report["overall"]
            writer.writerow(
                {
                    "class_id": "",
                    "name": "ALL (COCO mean)",
                    "AP": o["AP"],
                    "AP50": o["AP50"],
                    "AP75": o["AP75"],
                    "Precision": o["Precision"],
                    "Recall": o["Recall"],
                    "F1": o["F1"],
                }
            )


# --------------------------------------------------------------------------
# Standalone CLI -- evaluate a saved YOLO best.pt directly
# --------------------------------------------------------------------------
def _predict_with_model(model_path, data_yaml_path, dataset_root=None, split="val"):
    """Run the saved YOLO model on every image in the requested split."""
    data_yaml_path = Path(data_yaml_path)

    with open(data_yaml_path) as f:
        cfg = yaml.safe_load(f)

    root = (
        Path(dataset_root)
        if dataset_root
        else Path(cfg.get("path", data_yaml_path.parent.resolve())).resolve()
    )

    model = YOLO(str(model_path))
    predictions = []

    for images_dir, _labels_dir in resolve_split_dirs(data_yaml_path, split):
        images_dir = Path(images_dir)
        if not images_dir.exists():
            continue

        image_paths = sorted(
            p
            for p in images_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        )

        if not image_paths:
            continue

        results = model.predict(
            source=[str(p) for p in image_paths],
            conf=0.001,  # retain low-confidence detections for AP ranking
            max_det=100,  # COCO maxDets=100
            save=False,
            verbose=False,
        )

        for image_path, result in zip(image_paths, results):
            image_id = compute_image_id(image_path, root)

            if result.boxes is None or len(result.boxes) == 0:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()

            for box, cls, confidence in zip(boxes, classes, confidences):
                predictions.append(
                    {
                        "image_id": image_id,
                        "class_id": int(cls),
                        "bbox": box.tolist(),
                        "score": float(confidence),
                    }
                )

    return predictions


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate a saved YOLO best.pt directly on the validation "
        "split. Computes COCO AP@[.50:.95], AP50, AP75, Precision, Recall, F1, "
        "and a confusion matrix."
    )
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--model", required=True, help="Path to the saved best.pt model")
    parser.add_argument("--split", default="val", help="Dataset split to evaluate (default: val)")
    parser.add_argument(
        "--dataset-root",
        help="Shared dataset root used to create unique relative-path image IDs. "
        "Defaults to the path in data.yaml.",
    )
    parser.add_argument(
        "--iou-thresh",
        type=float,
        default=0.5,
        help="IoU threshold for the confusion matrix (default: 0.5)",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Where to write evaluation.json, evaluation.csv, and confusion_matrix.csv",
    )

    args = parser.parse_args()

    data_yaml = Path(args.data)
    model_path = Path(args.model)

    if not data_yaml.exists():
        raise FileNotFoundError(f"Data YAML not found: {data_yaml}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)

    dataset_root = Path(
        args.dataset_root if args.dataset_root else cfg.get("path", data_yaml.parent.resolve())
    ).resolve()

    print("=" * 60)
    print("  MILITARY ASSET DETECTION -- EVALUATION")
    print("=" * 60)
    print(f"  Model        : {model_path.resolve()}")
    print(f"  Dataset      : {data_yaml.resolve()}")
    print(f"  Split        : {args.split}")
    print(f"  Dataset root : {dataset_root}")
    print("=" * 60)

    ground_truth = load_dataset_split(
        data_yaml,
        split=args.split,
        dataset_root=dataset_root,
        is_prediction=False,
    )

    print(f"\n  Ground-truth boxes: {len(ground_truth)}")

    predictions = _predict_with_model(
        model_path,
        data_yaml,
        dataset_root=dataset_root,
        split=args.split,
    )

    print(f"  Prediction boxes  : {len(predictions)}")

    class_names = load_class_names(data_yaml)

    evaluator = DetectionEvaluator(ground_truth, predictions, class_names)
    report = evaluator.evaluate()
    evaluator.print_report(report)

    cm = evaluator.confusion_matrix(args.iou_thresh)
    evaluator.print_confusion_matrix(cm)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluator.save_json(report, str(output_dir / "evaluation.json"))
    evaluator.save_csv(report, str(output_dir / "evaluation.csv"))
    evaluator.save_confusion_matrix_csv(
        cm,
        str(output_dir / "confusion_matrix.csv"),
    )

    print(f"\n[OK] Reports written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
