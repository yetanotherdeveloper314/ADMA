"""
Model-agnostic evaluation for ADMA object detectors.

Ground truth is read from a YOLO-format dataset described by a data.yaml
(a single Roboflow export or the combined dataset built by
``scripts/train.py --dataset all``). Predictions can come from any model
through one of three entry points:

* ``predict_with(predictor, dataset)`` runs any callable that maps an image
  path to detection records, and measures per-image latency.
  ``UltralyticsPredictor`` wraps Ultralytics weights (YOLOv8, YOLO11, ...).
* ``load_yolo_predictions(directory, dataset)`` reads YOLO txt files with a
  confidence column (Ultralytics ``save_txt=True, save_conf=True``).
* ``load_json_predictions(path, dataset)`` reads a JSON list of records.

A detection record is the same shape ``MilitaryAssetDetector.detect()``
returns, with ``bbox`` in absolute pixels::

    {"class": "tank", "confidence": 0.91, "bbox": [x1, y1, x2, y2]}

``class_id`` may replace ``class`` and ``score`` may replace ``confidence``.

Metrics follow COCO conventions: AP uses 101-point interpolated precision,
is averaged over IoU thresholds 0.50:0.05:0.95, and is then averaged over
classes that have ground truth. Precision, recall, F1 and the confusion
matrix are reported at an operating confidence threshold, because at the
very low confidence used for AP they mostly measure noise.

Boxes are stored as normalized [x1, y1, x2, y2]. IoU is unchanged by
per-axis scaling, so YOLO labels are compared without reading image sizes.
"""

from __future__ import annotations

import csv
import json
import os
import time
import warnings
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
IOU_THRESHOLDS = np.round(np.linspace(0.5, 0.95, 10), 2)
RECALL_THRESHOLDS = np.linspace(0.0, 1.0, 101)
CONFIDENCE_GRID = np.linspace(0.0, 1.0, 1001)
DEFAULT_MAX_DET = 100

Predictor = Callable[[Path], Iterable[Mapping]]


# ------------------------------------------------------------------
# Data model
# ------------------------------------------------------------------


@dataclass(frozen=True)
class Detection:
    """One box in normalized [x1, y1, x2, y2] coordinates."""

    image_id: str
    class_id: int
    bbox: tuple[float, float, float, float]
    score: float = 1.0


@dataclass
class EvalDataset:
    """Ground truth for one split. Image ids are paths relative to ``root``, without suffix."""

    class_names: dict[int, str]
    images: dict[str, Path]
    ground_truth: list[Detection]
    root: Path


@dataclass
class LatencyStats:
    images: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    fps: float

    @classmethod
    def from_seconds(cls, samples: list[float]) -> LatencyStats | None:
        if not samples:
            return None
        ms = np.asarray(samples) * 1000.0
        mean = float(ms.mean())
        return cls(
            images=len(samples),
            mean_ms=mean,
            median_ms=float(np.median(ms)),
            p95_ms=float(np.percentile(ms, 95)),
            fps=1000.0 / mean if mean > 0 else 0.0,
        )


@dataclass
class ClassMetrics:
    class_id: int
    name: str
    num_gt: int
    num_pred: int
    ap50_95: float | None
    ap50: float | None
    ap75: float | None
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


@dataclass
class EvaluationReport:
    per_class: list[ClassMetrics]
    map50_95: float
    map50: float
    map75: float
    precision: float
    recall: float
    f1: float
    best_f1: float
    best_f1_confidence: float
    confusion_labels: list[str]
    confusion_matrix: np.ndarray
    conf_threshold: float
    iou_threshold: float
    max_det: int
    num_images: int
    num_gt: int
    num_pred: int
    latency: LatencyStats | None = field(default=None)

    def to_dict(self) -> dict:
        return {
            "summary": {
                "mAP50-95": self.map50_95,
                "mAP50": self.map50,
                "mAP75": self.map75,
                "precision": self.precision,
                "recall": self.recall,
                "f1": self.f1,
                "best_f1": self.best_f1,
                "best_f1_confidence": self.best_f1_confidence,
            },
            "settings": {
                "conf_threshold": self.conf_threshold,
                "iou_threshold": self.iou_threshold,
                "max_det": self.max_det,
            },
            "counts": {
                "images": self.num_images,
                "ground_truth": self.num_gt,
                "predictions": self.num_pred,
            },
            "latency": asdict(self.latency) if self.latency else None,
            "per_class": [asdict(m) for m in self.per_class],
            "confusion_matrix": {
                "labels": self.confusion_labels,
                "matrix": self.confusion_matrix.tolist(),
            },
        }


# ------------------------------------------------------------------
# Ground truth
# ------------------------------------------------------------------


def load_class_names(data_yaml: str | Path) -> dict[int, str]:
    return _parse_names(_read_yaml(Path(data_yaml)))


def load_dataset(data_yaml: str | Path, split: str = "val") -> EvalDataset:
    """Load every image and YOLO label of ``split`` from a data.yaml."""
    yaml_path = Path(data_yaml)
    cfg = _read_yaml(yaml_path)
    class_names = _parse_names(cfg)
    root = _dataset_root(cfg, yaml_path)

    image_paths: dict[Path, None] = {}
    for source in _split_sources(cfg, root, split):
        if not source.exists():
            warnings.warn(f"Skipping missing '{split}' source: {source}", stacklevel=2)
            continue
        image_paths.update(dict.fromkeys(_list_images(source)))
    if not image_paths:
        raise FileNotFoundError(f"No images found for split '{split}' in {yaml_path}")

    id_root = Path(os.path.commonpath([p.parent for p in image_paths]))
    images: dict[str, Path] = {}
    ground_truth: list[Detection] = []
    for path in image_paths:
        image_id = path.relative_to(id_root).with_suffix("").as_posix()
        if image_id in images:
            raise ValueError(
                f"Images {images[image_id]} and {path} map to the same id {image_id!r}"
            )
        images[image_id] = path
        label_path = image_to_label_path(path)
        if label_path.exists():
            ground_truth.extend(_read_label_file(label_path, image_id, class_names, False))

    return EvalDataset(class_names, images, ground_truth, id_root)


def image_to_label_path(image_path: Path) -> Path:
    """Swap the last ``images`` directory for ``labels``, as Ultralytics does."""
    parts = list(image_path.parts)
    for i in range(len(parts) - 2, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Data YAML not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _parse_names(cfg: dict) -> dict[int, str]:
    names = cfg.get("names")
    if not names:
        raise ValueError("data.yaml has no 'names' entry")
    if isinstance(names, list):
        return dict(enumerate(str(n) for n in names))
    return {int(k): str(v) for k, v in names.items()}


def _dataset_root(cfg: dict, yaml_path: Path) -> Path:
    path = cfg.get("path")
    if not path:
        return yaml_path.parent.resolve()
    path = Path(path)
    return path if path.is_absolute() else (yaml_path.parent / path).resolve()


def _split_sources(cfg: dict, root: Path, split: str) -> list[Path]:
    if not cfg.get(split):
        raise ValueError(f"data.yaml has no '{split}' split")
    entries = cfg[split] if isinstance(cfg[split], list) else [cfg[split]]
    sources = []
    for entry in entries:
        entry = str(entry)
        candidate = (root / entry).resolve()
        # Roboflow exports write '../valid/images'; Ultralytics retries without the '../'.
        if not candidate.exists() and entry.startswith("../"):
            candidate = (root / entry[3:]).resolve()
        sources.append(candidate)
    return sources


def _list_images(source: Path) -> list[Path]:
    if source.is_dir():
        return sorted(
            p.resolve()
            for p in source.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
    if source.suffix == ".txt":
        paths = []
        for line in source.read_text().splitlines():
            line = line.strip()
            if line:
                p = Path(line)
                paths.append((p if p.is_absolute() else source.parent / p).resolve())
        return [p for p in paths if p.suffix.lower() in IMAGE_EXTENSIONS]
    raise ValueError(f"Split source must be a directory or a .txt image list: {source}")


def _read_label_file(
    path: Path, image_id: str, class_names: dict[int, str], with_scores: bool
) -> list[Detection]:
    detections = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        values = line.split()
        if not values:
            continue
        where = f"{path}:{line_no}"
        try:
            numbers = [float(v) for v in values]
        except ValueError as e:
            raise ValueError(f"{where}: non-numeric value in {line!r}") from e

        score = 1.0
        if with_scores:
            if len(numbers) != 6:
                raise ValueError(
                    f"{where}: expected 'class xc yc w h confidence' "
                    "(save predictions with save_conf=True)"
                )
            bbox = _xywh_to_xyxy(numbers[1:5])
            score = numbers[5]
        elif len(numbers) == 5:
            bbox = _xywh_to_xyxy(numbers[1:5])
        elif len(numbers) >= 7 and len(numbers) % 2 == 1:
            xs, ys = numbers[1::2], numbers[2::2]
            bbox = (min(xs), min(ys), max(xs), max(ys))
        else:
            raise ValueError(f"{where}: expected a box (5 values) or a polygon (odd count >= 7)")

        class_id = int(numbers[0])
        if class_id not in class_names:
            raise ValueError(f"{where}: class id {class_id} is not in data.yaml names")
        detections.append(Detection(image_id, class_id, bbox, score))
    return detections


def _xywh_to_xyxy(xywh: list[float]) -> tuple[float, float, float, float]:
    xc, yc, w, h = xywh
    return (xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2)


# ------------------------------------------------------------------
# Predictions
# ------------------------------------------------------------------


class UltralyticsPredictor:
    """Predictor for Ultralytics detection weights (YOLOv8, YOLO11, ...)."""

    def __init__(
        self,
        weights: str | Path,
        conf: float = 0.001,
        iou: float = 0.7,
        imgsz: int = 640,
        max_det: int = 300,
        device: str | None = None,
    ):
        # Imported here so evaluating saved predictions does not require torch.
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.names = self.model.names
        self.options = {
            "conf": conf,
            "iou": iou,
            "imgsz": imgsz,
            "max_det": max_det,
            "device": device,
            "verbose": False,
        }

    def __call__(self, image_path: Path) -> list[dict]:
        boxes = self.model.predict(str(image_path), **self.options)[0].boxes
        return [
            {"class": self.names[int(cls)], "confidence": conf, "bbox": xyxy}
            for xyxy, cls, conf in zip(boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist())
        ]


def predict_with(
    predictor: Predictor,
    dataset: EvalDataset,
    class_map: Mapping[str, str | None] | None = None,
    warmup: int = 1,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[list[Detection], LatencyStats | None]:
    """Run ``predictor`` on every image of ``dataset``, timing each call."""
    resolver = _ClassResolver(dataset.class_names, class_map)
    items = list(dataset.images.items())
    for _, path in items[:warmup]:
        list(predictor(path))

    predictions: list[Detection] = []
    timings: list[float] = []
    for i, (image_id, path) in enumerate(items, start=1):
        start = time.perf_counter()
        records = list(predictor(path))
        timings.append(time.perf_counter() - start)
        size = image_size(path)
        for record in records:
            detection = _record_to_detection(record, image_id, size, resolver)
            if detection is not None:
                predictions.append(detection)
        if progress is not None:
            progress(i, len(items))

    resolver.warn_dropped()
    return predictions, LatencyStats.from_seconds(timings)


def load_yolo_predictions(pred_dir: str | Path, dataset: EvalDataset) -> list[Detection]:
    """
    Read ``class xc yc w h confidence`` txt files. Files may mirror the image
    ids (``tank_images/img1.txt``) or be flat by stem (``img1.txt``) as
    Ultralytics writes them. Class ids must follow the data.yaml order.
    """
    pred_dir = Path(pred_dir)
    if not pred_dir.is_dir():
        raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")

    ids_by_stem: dict[str, list[str]] = defaultdict(list)
    for image_id in dataset.images:
        ids_by_stem[Path(image_id).name].append(image_id)

    used: set[Path] = set()
    predictions: list[Detection] = []
    for image_id in dataset.images:
        stem = Path(image_id).name
        path = pred_dir / f"{image_id}.txt"
        if not path.exists():
            path = pred_dir / f"{stem}.txt"
            if not path.exists():
                continue
            if len(ids_by_stem[stem]) > 1:
                raise ValueError(
                    f"{path} is ambiguous: images {ids_by_stem[stem]} share the stem {stem!r}. "
                    "Mirror the image folder structure in the prediction directory."
                )
        used.add(path.resolve())
        predictions.extend(_read_label_file(path, image_id, dataset.class_names, True))

    unused = [p for p in pred_dir.rglob("*.txt") if p.resolve() not in used]
    if unused:
        warnings.warn(
            f"{len(unused)} prediction file(s) in {pred_dir} match no image in the split, "
            f"e.g. {unused[0]}",
            stacklevel=2,
        )
    return predictions


def load_json_predictions(
    path: str | Path,
    dataset: EvalDataset,
    class_map: Mapping[str, str | None] | None = None,
) -> list[Detection]:
    """
    Read pixel-space detection records from JSON, either a list of records
    with an ``image`` key or a ``{image: [records]}`` mapping. ``image`` may
    be the image id, a path relative to the dataset root, an absolute path,
    or a filename when its stem is unique in the split.
    """
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        items = [(image, record) for image, records in data.items() for record in records]
    elif isinstance(data, list):
        try:
            items = [(record["image"], record) for record in data]
        except (KeyError, TypeError) as e:
            raise ValueError("Every JSON prediction record needs an 'image' key") from e
    else:
        raise ValueError("Prediction JSON must be a list of records or an {image: records} map")

    lookup = _ImageLookup(dataset)
    resolver = _ClassResolver(dataset.class_names, class_map)
    sizes: dict[str, tuple[int, int]] = {}
    predictions: list[Detection] = []
    for image, record in items:
        image_id = lookup(image)
        if image_id not in sizes:
            sizes[image_id] = image_size(dataset.images[image_id])
        detection = _record_to_detection(record, image_id, sizes[image_id], resolver)
        if detection is not None:
            predictions.append(detection)

    resolver.warn_dropped()
    return predictions


def image_size(path: Path) -> tuple[int, int]:
    """(width, height) as decoders that honor EXIF orientation see the image."""
    with Image.open(path) as image:
        width, height = image.size
        # EXIF orientations 5-8 rotate the image by 90 degrees.
        if image.getexif().get(0x0112) in {5, 6, 7, 8}:
            width, height = height, width
    return width, height


class _ImageLookup:
    def __init__(self, dataset: EvalDataset):
        self.ids = set(dataset.images)
        self.root = dataset.root
        self.by_path = {str(p): image_id for image_id, p in dataset.images.items()}
        stems: dict[str, list[str]] = defaultdict(list)
        for image_id in dataset.images:
            stems[Path(image_id).name].append(image_id)
        self.by_stem = {s: ids[0] for s, ids in stems.items() if len(ids) == 1}

    def __call__(self, key: str) -> str:
        path = Path(str(key))
        if path.is_absolute():
            resolved = str(path.resolve())
            if resolved in self.by_path:
                return self.by_path[resolved]
            try:
                path = path.resolve().relative_to(self.root)
            except ValueError:
                pass
        for candidate in (path.as_posix(), path.with_suffix("").as_posix()):
            if candidate in self.ids:
                return candidate
        if path.stem in self.by_stem:
            return self.by_stem[path.stem]
        raise ValueError(f"Prediction image {key!r} does not match any image in the split")


class _ClassResolver:
    """
    Maps a record's class to a dataset class id. Model class names must match
    data.yaml names unless renamed in ``class_map`` (``None`` drops a class).
    When a map is given, unknown model classes are dropped instead of raising.
    """

    def __init__(self, class_names: dict[int, str], class_map: Mapping[str, str | None] | None):
        self.class_names = class_names
        self.name_to_id = {name: class_id for class_id, name in class_names.items()}
        self.class_map = dict(class_map) if class_map else None
        self.dropped: set[str] = set()
        for source, target in (self.class_map or {}).items():
            if target is not None and target not in self.name_to_id:
                raise ValueError(f"Class map sends {source!r} to unknown dataset class {target!r}")

    def __call__(self, record: Mapping) -> int | None:
        if "class" in record:
            name = str(record["class"])
            if self.class_map is not None and name in self.class_map:
                target = self.class_map[name]
                return None if target is None else self.name_to_id[target]
            if name in self.name_to_id:
                return self.name_to_id[name]
            if self.class_map is not None:
                self.dropped.add(name)
                return None
            raise ValueError(
                f"Model class {name!r} is not a dataset class ({sorted(self.name_to_id)}). "
                "Provide a class map to rename or drop model classes."
            )
        if "class_id" in record:
            class_id = int(record["class_id"])
            if class_id not in self.class_names:
                raise ValueError(f"Class id {class_id} is not in data.yaml names")
            return class_id
        raise ValueError(f"Detection record needs a 'class' or 'class_id' key: {dict(record)}")

    def warn_dropped(self) -> None:
        if self.dropped:
            warnings.warn(
                f"Dropped predictions for classes not in the class map: {sorted(self.dropped)}",
                stacklevel=3,
            )


def _record_to_detection(
    record: Mapping, image_id: str, size: tuple[int, int], resolver: _ClassResolver
) -> Detection | None:
    class_id = resolver(record)
    if class_id is None:
        return None
    score = record.get("confidence", record.get("score"))
    if score is None:
        raise ValueError(f"Detection record needs a 'confidence' or 'score' key: {dict(record)}")
    bbox = [float(v) for v in record["bbox"]]
    if len(bbox) != 4:
        raise ValueError(f"bbox must be [x1, y1, x2, y2] in pixels: {record['bbox']}")
    width, height = size
    x1, y1, x2, y2 = bbox
    return Detection(
        image_id, class_id, (x1 / width, y1 / height, x2 / width, y2 / height), float(score)
    )


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------


def box_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of (N, 4) and (M, 4) xyxy boxes, shape (N, M)."""
    a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(axis=2)
    area_a = np.clip(a[:, 2:] - a[:, :2], 0, None).prod(axis=1)
    area_b = np.clip(b[:, 2:] - b[:, :2], 0, None).prod(axis=1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def match_detections(ious: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """
    COCO greedy matching for one image and class. ``ious`` is (preds, gts)
    with predictions sorted by descending score; each prediction takes the
    highest-IoU ground truth not yet taken at that threshold. Returns a
    (preds, thresholds) boolean true-positive matrix.
    """
    num_preds, num_gt = ious.shape
    tp = np.zeros((num_preds, len(thresholds)), dtype=bool)
    if num_preds == 0 or num_gt == 0:
        return tp
    taken = np.zeros((len(thresholds), num_gt), dtype=bool)
    rows = np.arange(len(thresholds))
    for i in range(num_preds):
        candidates = np.where(taken, -1.0, ious[i][None, :])
        best = candidates.argmax(axis=1)
        hit = candidates[rows, best] >= thresholds
        tp[i] = hit
        taken[rows[hit], best[hit]] = True
    return tp


def average_precision(tp: np.ndarray, num_gt: int) -> float | None:
    """COCO 101-point AP from a true-positive vector sorted by descending score."""
    if num_gt == 0:
        return None
    if tp.size == 0:
        return 0.0
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(~tp)
    recall = tp_cum / num_gt
    precision = tp_cum / (tp_cum + fp_cum)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    idx = np.searchsorted(recall, RECALL_THRESHOLDS, side="left")
    sampled = np.where(idx < precision.size, precision[np.minimum(idx, precision.size - 1)], 0.0)
    return float(sampled.mean())


def precision_recall_f1(tp: int, num_pred: int, num_gt: int) -> tuple[float, float, float]:
    precision = tp / num_pred if num_pred else 0.0
    recall = tp / num_gt if num_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def evaluate(
    dataset: EvalDataset,
    predictions: list[Detection],
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.5,
    max_det: int = DEFAULT_MAX_DET,
    latency: LatencyStats | None = None,
) -> EvaluationReport:
    """
    Score ``predictions`` against ``dataset``. At most ``max_det``
    highest-scoring predictions per image are kept. AP uses every kept
    prediction; precision/recall/F1 and the confusion matrix use those with
    score >= ``conf_threshold``, matched at ``iou_threshold``.
    """
    _validate(dataset, predictions)
    predictions = _top_k_per_image(predictions, max_det)
    thresholds = np.append(IOU_THRESHOLDS, iou_threshold)
    col50 = int(np.flatnonzero(IOU_THRESHOLDS == 0.5)[0])
    col75 = int(np.flatnonzero(IOU_THRESHOLDS == 0.75)[0])

    gt_index: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for g in dataset.ground_truth:
        gt_index[g.class_id][g.image_id].append(g.bbox)
    pred_index: dict[int, dict[str, list[Detection]]] = defaultdict(lambda: defaultdict(list))
    for p in predictions:
        pred_index[p.class_id][p.image_id].append(p)

    per_class: list[ClassMetrics] = []
    f1_curves: list[np.ndarray] = []
    for class_id, name in sorted(dataset.class_names.items()):
        class_gt = gt_index.get(class_id, {})
        num_gt = sum(len(boxes) for boxes in class_gt.values())

        scores_parts, tp_parts = [], []
        for image_id, preds in pred_index.get(class_id, {}).items():
            preds = sorted(preds, key=lambda d: -d.score)
            ious = box_iou([d.bbox for d in preds], class_gt.get(image_id, []))
            tp_parts.append(match_detections(ious, thresholds))
            scores_parts.append([d.score for d in preds])
        if tp_parts:
            scores = np.concatenate(scores_parts)
            order = np.argsort(-scores, kind="mergesort")
            scores, tp = scores[order], np.concatenate(tp_parts)[order]
        else:
            scores, tp = np.empty(0), np.zeros((0, len(thresholds)), dtype=bool)

        aps = [average_precision(tp[:, t], num_gt) for t in range(len(IOU_THRESHOLDS))]
        has_gt = num_gt > 0

        tp_cum = np.concatenate([[0], np.cumsum(tp[:, -1])])
        num_pred = int(np.searchsorted(-scores, -conf_threshold, side="right"))
        tp_count = int(tp_cum[num_pred])
        precision, recall, f1 = precision_recall_f1(tp_count, num_pred, num_gt)
        if has_gt:
            f1_curves.append(_f1_curve(scores, tp_cum, num_gt))

        per_class.append(
            ClassMetrics(
                class_id=class_id,
                name=name,
                num_gt=num_gt,
                num_pred=num_pred,
                ap50_95=float(np.mean(aps)) if has_gt else None,
                ap50=aps[col50],
                ap75=aps[col75],
                precision=precision,
                recall=recall,
                f1=f1,
                tp=tp_count,
                fp=num_pred - tp_count,
                fn=num_gt - tp_count,
            )
        )

    scored = [m for m in per_class if m.num_gt > 0]

    def mean(attr: str) -> float:
        return float(np.mean([getattr(m, attr) for m in scored])) if scored else 0.0

    if f1_curves:
        macro_f1 = np.mean(f1_curves, axis=0)
        best = int(macro_f1.argmax())
        best_f1, best_conf = float(macro_f1[best]), float(CONFIDENCE_GRID[best])
    else:
        best_f1, best_conf = 0.0, 0.0

    labels, matrix = confusion_matrix(dataset, predictions, conf_threshold, iou_threshold)
    return EvaluationReport(
        per_class=per_class,
        map50_95=mean("ap50_95"),
        map50=mean("ap50"),
        map75=mean("ap75"),
        precision=mean("precision"),
        recall=mean("recall"),
        f1=mean("f1"),
        best_f1=best_f1,
        best_f1_confidence=best_conf,
        confusion_labels=labels,
        confusion_matrix=matrix,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        max_det=max_det,
        num_images=len(dataset.images),
        num_gt=len(dataset.ground_truth),
        num_pred=len(predictions),
        latency=latency,
    )


def confusion_matrix(
    dataset: EvalDataset,
    predictions: list[Detection],
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.5,
) -> tuple[list[str], np.ndarray]:
    """
    Rows are ground-truth classes, columns predicted classes, and the last
    row/column is background. Boxes are paired class-agnostically in
    descending IoU order, so a well-placed box with the wrong class counts as
    a confusion rather than a miss plus a false positive.
    """
    class_ids = sorted(dataset.class_names)
    index = {c: i for i, c in enumerate(class_ids)}
    background = len(class_ids)
    matrix = np.zeros((background + 1, background + 1), dtype=int)

    gt_by_image: dict[str, list[Detection]] = defaultdict(list)
    for g in dataset.ground_truth:
        gt_by_image[g.image_id].append(g)
    pred_by_image: dict[str, list[Detection]] = defaultdict(list)
    for p in predictions:
        if p.score >= conf_threshold:
            pred_by_image[p.image_id].append(p)

    for image_id in gt_by_image.keys() | pred_by_image.keys():
        gts, preds = gt_by_image.get(image_id, []), pred_by_image.get(image_id, [])
        ious = box_iou([p.bbox for p in preds], [g.bbox for g in gts])
        pairs = np.argwhere(ious >= iou_threshold)
        pairs = pairs[np.argsort(-ious[pairs[:, 0], pairs[:, 1]], kind="stable")]
        used_preds, used_gts = set(), set()
        for p, g in pairs:
            if p in used_preds or g in used_gts:
                continue
            used_preds.add(p)
            used_gts.add(g)
            matrix[index[gts[g].class_id], index[preds[p].class_id]] += 1
        for g, gt in enumerate(gts):
            if g not in used_gts:
                matrix[index[gt.class_id], background] += 1
        for p, pred in enumerate(preds):
            if p not in used_preds:
                matrix[background, index[pred.class_id]] += 1

    labels = [dataset.class_names[c] for c in class_ids] + ["background"]
    return labels, matrix


def _validate(dataset: EvalDataset, predictions: list[Detection]) -> None:
    unknown_images = {p.image_id for p in predictions} - dataset.images.keys()
    if unknown_images:
        example = sorted(unknown_images)[0]
        raise ValueError(
            f"{len(unknown_images)} prediction image id(s) are not in the dataset, e.g. {example!r}"
        )
    unknown_classes = {p.class_id for p in predictions} - dataset.class_names.keys()
    if unknown_classes:
        raise ValueError(f"Prediction class ids not in data.yaml names: {sorted(unknown_classes)}")


def _top_k_per_image(predictions: list[Detection], k: int) -> list[Detection]:
    by_image: dict[str, list[Detection]] = defaultdict(list)
    for p in predictions:
        by_image[p.image_id].append(p)
    kept = []
    for preds in by_image.values():
        kept.extend(sorted(preds, key=lambda d: -d.score)[:k])
    return kept


def _f1_curve(scores: np.ndarray, tp_cum: np.ndarray, num_gt: int) -> np.ndarray:
    """F1 at every confidence in CONFIDENCE_GRID; ``scores`` sorted descending."""
    counts = np.searchsorted(-scores, -CONFIDENCE_GRID, side="right")
    tps = tp_cum[counts]
    precision = np.divide(tps, counts, out=np.zeros(len(counts)), where=counts > 0)
    recall = tps / num_gt
    total = precision + recall
    return np.divide(2 * precision * recall, total, out=np.zeros(len(counts)), where=total > 0)


# ------------------------------------------------------------------
# Reporting
# ------------------------------------------------------------------


def format_report(report: EvaluationReport, top_confusions: int = 5) -> str:
    def fmt(value: float | None) -> str:
        return "-" if value is None else f"{value:.3f}"

    header = (
        f"{'Class':<22}{'GT':>7}{'Pred':>7}{'mAP50-95':>10}{'AP50':>8}{'AP75':>8}"
        f"{'P':>8}{'R':>8}{'F1':>8}"
    )
    lines = [header, "-" * len(header)]
    for m in report.per_class:
        lines.append(
            f"{m.name[:21]:<22}{m.num_gt:>7}{m.num_pred:>7}{fmt(m.ap50_95):>10}"
            f"{fmt(m.ap50):>8}{fmt(m.ap75):>8}{m.precision:>8.3f}{m.recall:>8.3f}{m.f1:>8.3f}"
        )
    lines.append("-" * len(header))
    num_pred = sum(m.num_pred for m in report.per_class)
    lines.append(
        f"{'ALL (class mean)':<22}{report.num_gt:>7}{num_pred:>7}{report.map50_95:>10.3f}"
        f"{report.map50:>8.3f}{report.map75:>8.3f}{report.precision:>8.3f}"
        f"{report.recall:>8.3f}{report.f1:>8.3f}"
    )
    lines += [
        "",
        f"  P/R/F1 and confusion matrix at confidence >= {report.conf_threshold:g}, "
        f"IoU >= {report.iou_threshold:g}",
        f"  Best mean F1 {report.best_f1:.3f} at confidence {report.best_f1_confidence:.3f}",
    ]

    matrix, labels = report.confusion_matrix, report.confusion_labels
    off_diagonal = [
        (int(matrix[r, c]), labels[r], labels[c])
        for r in range(len(labels))
        for c in range(len(labels))
        if r != c and matrix[r, c] > 0
    ]
    off_diagonal.sort(reverse=True)
    if off_diagonal and top_confusions:
        lines += ["", "  Top confusions (actual -> predicted):"]
        lines += [f"    {a} -> {p}: {n}" for n, a, p in off_diagonal[:top_confusions]]

    if report.latency:
        lat = report.latency
        lines += [
            "",
            f"  Latency over {lat.images} images: mean {lat.mean_ms:.1f} ms, "
            f"median {lat.median_ms:.1f} ms, p95 {lat.p95_ms:.1f} ms ({lat.fps:.1f} FPS)",
        ]
    return "\n".join(lines)


def save_report(
    report: EvaluationReport, output_dir: str | Path, metadata: Mapping | None = None
) -> dict[str, Path]:
    """Write metrics.json, per_class.csv and confusion_matrix.csv to ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "metrics": output_dir / "metrics.json",
        "per_class": output_dir / "per_class.csv",
        "confusion_matrix": output_dir / "confusion_matrix.csv",
    }

    payload = {"metadata": dict(metadata or {}), **report.to_dict()}
    paths["metrics"].write_text(json.dumps(payload, indent=2) + "\n")

    with open(paths["per_class"], "w", newline="") as f:
        fields = list(ClassMetrics.__dataclass_fields__)
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for m in report.per_class:
            writer.writerow(asdict(m))

    with open(paths["confusion_matrix"], "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["actual \\ predicted", *report.confusion_labels])
        for label, row in zip(report.confusion_labels, report.confusion_matrix):
            writer.writerow([label, *row.tolist()])

    return paths
