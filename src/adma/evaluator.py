"""
Model-Agnostic Object Detection Evaluator
==========================================

Computes Precision, Recall, F1, and mAP50-95 (COCO-style, averaged over
IoU thresholds 0.50:0.05:0.95) from any set of predictions and ground
truth boxes, regardless of which model produced the predictions.

INPUT FORMAT (internal, model-agnostic)
----------------------------------------
Ground truth : List[dict] with keys:
    image_id : str | int
    class_id : int
    bbox     : [x1, y1, x2, y2]   (absolute pixel coords)

Predictions  : List[dict] with keys:
    image_id : str | int
    class_id : int
    bbox     : [x1, y1, x2, y2]
    score    : float              (confidence, 0-1)

Loader provided for YOLO-format datasets (load_yolo). For a different
detector's raw output, write a small adapter (see predictions_from_detector)
that maps it into the same list-of-dicts shape -- everything else works
unchanged.

USAGE
-----
    from detection_evaluator import DetectionEvaluator, load_yolo, load_class_names

    class_names = load_class_names("data.yaml")
    ground_truth = load_yolo("labels/val", "images/val")
    predictions = predictions_from_detector(detector_results)

    evaluator = DetectionEvaluator(ground_truth, predictions, class_names)
    report = evaluator.evaluate()
    evaluator.print_report(report)
    evaluator.save_json(report, "evaluation.json")
    evaluator.save_csv(report, "evaluation.csv")
"""

from collections import defaultdict
import csv
import json
import os
import cv2
import numpy as np
import yaml


# --------------------------------------------------------------------------
# IoU utility
# --------------------------------------------------------------------------
def compute_iou(box_a, box_b):
    """IoU between two [x1,y1,x2,y2] boxes."""
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


# --------------------------------------------------------------------------
# Class names loader (from data.yaml)
# --------------------------------------------------------------------------
def load_class_names(data_yaml_path):
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
    with open(data_yaml_path, "r") as f:
        data = yaml.safe_load(f)

    names = data["names"]
    if isinstance(names, list):
        return {i: n for i, n in enumerate(names)}
    return {int(k): v for k, v in names.items()}


# --------------------------------------------------------------------------
# YOLO loader (model-agnostic front door for ground truth / predictions)
# --------------------------------------------------------------------------
def load_yolo(labels_dir, images_dir, is_prediction=False):
    """
    Load ground truth or predictions from YOLO-style txt files.

    labels_dir   : directory containing one .txt file per image
                   (filename stem == image_id)
    images_dir   : directory containing the corresponding images
                   (used to read width/height via OpenCV for de-normalizing
                   the YOLO coords -- same stem, common image extensions)
    is_prediction: if True, expects a trailing confidence column:
                   "class x_center y_center w h confidence"
                   otherwise: "class x_center y_center w h"
    """
    IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    def _find_image(stem):
        for ext in IMG_EXTS:
            candidate = os.path.join(images_dir, stem + ext)
            if os.path.exists(candidate):
                return candidate
        return None

    out = []
    for fname in os.listdir(labels_dir):
        if not fname.endswith(".txt"):
            continue
        image_id = os.path.splitext(fname)[0]

        img_path = _find_image(image_id)
        if img_path is None:
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        H, W = img.shape[:2]

        with open(os.path.join(labels_dir, fname), "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cls = int(float(parts[0]))
                xc, yc, w, h = map(float, parts[1:5])
                x1 = (xc - w / 2) * W
                y1 = (yc - h / 2) * H
                x2 = (xc + w / 2) * W
                y2 = (yc + h / 2) * H

                entry = {"image_id": image_id, "class_id": cls, "bbox": [x1, y1, x2, y2]}
                if is_prediction:
                    entry["score"] = float(parts[5]) if len(parts) > 5 else 1.0
                out.append(entry)
    return out


# --------------------------------------------------------------------------
# Prediction adapter for MilitaryAssetDetector.detect() output
# --------------------------------------------------------------------------
def predictions_from_detector(results, class_name_to_id=None):
    """
    Convert MilitaryAssetDetector.detect() output into the evaluator's
    expected format.

    Expected input: a dict {image_id: [detections, ...]} or a list of
    per-image results, where each result has:
        {
            "image_id": ...,          # optional if results is already keyed by image_id
            "class": "tank",
            "confidence": 0.94,
            "bbox": [x1, y1, x2, y2],
        }

    class_name_to_id : optional {name: class_id} mapping. If omitted, class
                        ids are assigned in first-seen order of the class
                        names encountered (only safe if you don't also need
                        those ids to line up with a specific data.yaml --
                        pass class_name_to_id built from load_class_names()
                        to guarantee alignment).
    """
    if class_name_to_id is None:
        class_name_to_id = {}

    def _class_id(name):
        if name not in class_name_to_id:
            class_name_to_id[name] = len(class_name_to_id)
        return class_name_to_id[name]

    out = []

    if isinstance(results, dict):
        items = ((image_id, dets) for image_id, dets in results.items())
        for image_id, dets in items:
            for det in dets:
                out.append({
                    "image_id": image_id,
                    "class_id": _class_id(det["class"]),
                    "bbox": list(det["bbox"]),
                    "score": float(det["confidence"]),
                })
    else:
        for det in results:
            out.append({
                "image_id": det["image_id"],
                "class_id": _class_id(det["class"]),
                "bbox": list(det["bbox"]),
                "score": float(det["confidence"]),
            })

    return out


# --------------------------------------------------------------------------
# Core evaluator
# --------------------------------------------------------------------------
class DetectionEvaluator:
    def __init__(self, ground_truths, predictions, class_names=None):
        """
        ground_truths, predictions : lists of dicts (see module docstring)
        class_names : optional {class_id: name} for a readable report
                      (e.g. from load_class_names("data.yaml"))
        """
        self.gt = ground_truths
        self.preds = predictions
        self.class_names = class_names or {}
        self.classes = sorted(set(
            [g["class_id"] for g in ground_truths] +
            [p["class_id"] for p in predictions]
        ))
        self.iou_thresholds = np.round(np.arange(0.50, 1.00, 0.05), 2)  # 0.50..0.95

    # ---- matching at a single IoU threshold, single class -------------
    def _match_single_class(self, class_id, iou_thresh):
        gt_by_image = defaultdict(list)
        for g in self.gt:
            if g["class_id"] == class_id:
                gt_by_image[g["image_id"]].append(g["bbox"])

        class_preds = [p for p in self.preds if p["class_id"] == class_id]
        class_preds.sort(key=lambda p: p["score"], reverse=True)

        matched = {img: np.zeros(len(boxes), dtype=bool) for img, boxes in gt_by_image.items()}
        n_gt = sum(len(v) for v in gt_by_image.values())

        tp = np.zeros(len(class_preds))
        fp = np.zeros(len(class_preds))

        for i, pred in enumerate(class_preds):
            gt_boxes = gt_by_image.get(pred["image_id"], [])
            best_iou, best_j = 0.0, -1
            for j, gb in enumerate(gt_boxes):
                if matched[pred["image_id"]][j]:
                    continue
                iou = compute_iou(pred["bbox"], gb)
                if iou > best_iou:
                    best_iou, best_j = iou, j

            if best_iou >= iou_thresh and best_j >= 0:
                tp[i] = 1
                matched[pred["image_id"]][best_j] = True
            else:
                fp[i] = 1

        return tp, fp, n_gt

    # ---- AP for one class at one IoU threshold (area under PR curve) --
    def _average_precision(self, class_id, iou_thresh):
        tp, fp, n_gt = self._match_single_class(class_id, iou_thresh)
        if n_gt == 0:
            return None, 0.0, 0.0  # undefined, skip in mAP average

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recalls = tp_cum / (n_gt + 1e-12)
        precisions = tp_cum / (tp_cum + fp_cum + 1e-12)

        # monotonic envelope
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i + 1])

        recalls = np.concatenate(([0.0], recalls, [1.0]))
        precisions = np.concatenate(([precisions[0] if len(precisions) else 0.0],
                                      precisions, [0.0]))
        ap = float(np.sum(np.diff(recalls) * precisions[1:]))  # trapezoid-style AUC under PR curve

        final_p = precisions[-2] if len(precisions) > 1 else 0.0
        final_r = recalls[-2] if len(recalls) > 1 else 0.0
        return ap, final_p, final_r

    # ---- Precision / Recall / F1 at IoU 0.5, using best-score-per-box --
    def _prf_at_50(self, class_id):
        tp, fp, n_gt = self._match_single_class(class_id, 0.50)
        tp_sum, fp_sum = tp.sum(), fp.sum()
        fn_sum = n_gt - tp_sum

        precision = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0
        recall = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        return precision, recall, f1

    # ---- full report -----------------------------------------------------
    def evaluate(self):
        per_class = {}
        for c in self.classes:
            ap_per_iou = []
            for thr in self.iou_thresholds:
                ap, _, _ = self._average_precision(c, thr)
                if ap is not None:
                    ap_per_iou.append(ap)
            map_50_95 = float(np.mean(ap_per_iou)) if ap_per_iou else 0.0

            ap50, _, _ = self._average_precision(c, 0.50)
            precision, recall, f1 = self._prf_at_50(c)

            per_class[c] = {
                "name": self.class_names.get(c, str(c)),
                "AP50": ap50 if ap50 is not None else 0.0,
                "mAP50-95": map_50_95,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
            }

        overall = {
            "mAP50-95": float(np.mean([v["mAP50-95"] for v in per_class.values()])) if per_class else 0.0,
            "mAP50": float(np.mean([v["AP50"] for v in per_class.values()])) if per_class else 0.0,
            "Precision": float(np.mean([v["Precision"] for v in per_class.values()])) if per_class else 0.0,
            "Recall": float(np.mean([v["Recall"] for v in per_class.values()])) if per_class else 0.0,
            "F1": float(np.mean([v["F1"] for v in per_class.values()])) if per_class else 0.0,
        }
        return {"per_class": per_class, "overall": overall}

    @staticmethod
    def print_report(report):
        print(f"{'Class':<20}{'AP50':>8}{'mAP50-95':>12}{'Precision':>12}{'Recall':>10}{'F1':>8}")
        print("-" * 70)
        for c, v in report["per_class"].items():
            print(f"{v['name']:<20}{v['AP50']:>8.3f}{v['mAP50-95']:>12.3f}"
                  f"{v['Precision']:>12.3f}{v['Recall']:>10.3f}{v['F1']:>8.3f}")
        print("-" * 70)
        o = report["overall"]
        print(f"{'ALL (mean)':<20}{o['mAP50']:>8.3f}{o['mAP50-95']:>12.3f}"
              f"{o['Precision']:>12.3f}{o['Recall']:>10.3f}{o['F1']:>8.3f}")

    @staticmethod
    def save_json(report, path):
        """Write the full report (per-class + overall) to a JSON file."""
        # class_id keys must be strings for JSON
        serializable = {
            "per_class": {str(k): v for k, v in report["per_class"].items()},
            "overall": report["overall"],
        }
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)

    @staticmethod
    def save_csv(report, path):
        """Write the per-class breakdown (plus an ALL row) to a CSV file."""
        fieldnames = ["class_id", "name", "AP50", "mAP50-95", "Precision", "Recall", "F1"]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for class_id, v in report["per_class"].items():
                writer.writerow({
                    "class_id": class_id,
                    "name": v["name"],
                    "AP50": v["AP50"],
                    "mAP50-95": v["mAP50-95"],
                    "Precision": v["Precision"],
                    "Recall": v["Recall"],
                    "F1": v["F1"],
                })
            o = report["overall"]
            writer.writerow({
                "class_id": "",
                "name": "ALL (mean)",
                "AP50": o["mAP50"],
                "mAP50-95": o["mAP50-95"],
                "Precision": o["Precision"],
                "Recall": o["Recall"],
                "F1": o["F1"],
            })