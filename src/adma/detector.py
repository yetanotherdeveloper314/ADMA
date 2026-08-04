"""
ADMA Detection Engine

Uses a YOLOv8 model fine-tuned exclusively on military assets.
Only detects military objects, civilian items are never predicted
because the model is trained solely on military classes.
"""

from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from adma.config import DEFAULT_CONFIDENCE, MODELS_DIR


def discover_models() -> dict[str, Path]:
    """Scan the models directory and return a {name: path} mapping of trained models."""
    models: dict[str, Path] = {}
    mdir = Path(MODELS_DIR)
    if not mdir.exists():
        return models
    for pt in sorted(mdir.rglob("best.pt")):
        models[pt.parent.name] = pt
    return models


BBOX_COLORS = [
    (0, 0, 255),  # red
    (0, 165, 255),  # orange
    (0, 255, 255),  # yellow
    (0, 200, 0),  # green
    (255, 200, 0),  # cyan
    (255, 0, 0),  # blue
    (255, 0, 200),  # magenta
    (128, 0, 255),  # purple
]


class MilitaryAssetDetector:
    """Detect military assets in images using a fine-tuned YOLOv8 model."""

    def __init__(self, model_path: str, confidence: float = DEFAULT_CONFIDENCE):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found at '{model_path}'.\n"
                "Train one first:\n"
                "  python scripts/train.py --dataset <name> --epochs 100"
            )
        self.model = YOLO(str(path))
        self.confidence = confidence
        self.class_names = self.model.names

    def detect(self, image_source, confidence: float | None = None) -> dict:
        """Run military asset detection on a single image.

        Returns dict with keys: detections, annotated_image (numpy BGR),
        summary (str), and counts (dict).
        """
        conf = confidence if confidence is not None else self.confidence
        results = self.model(image_source, conf=conf, verbose=False)[0]

        detections = []
        for box in results.boxes:
            detections.append(
                {
                    "class": self.class_names[int(box.cls)],
                    "confidence": round(float(box.conf), 4),
                    "bbox": [round(v, 1) for v in box.xyxy[0].tolist()],
                }
            )

        counts = self._count_by_class(detections)
        summary = self._build_summary(detections, counts)
        annotated = self._draw_boxes(image_source, detections)

        return {
            "detections": detections,
            "annotated_image": annotated,
            "summary": summary,
            "counts": counts,
        }

    @staticmethod
    def _count_by_class(detections: list[dict]) -> dict:
        counts: dict[str, dict] = {}
        for d in detections:
            cls = d["class"]
            if cls not in counts:
                counts[cls] = {"count": 0, "confidences": []}
            counts[cls]["count"] += 1
            counts[cls]["confidences"].append(d["confidence"])
        return counts

    @staticmethod
    def _build_summary(detections: list[dict], counts: dict) -> str:
        if not detections:
            return "No military assets detected in this image."

        total = len(detections)
        lines = [f"Detected {total} military asset(s):\n"]
        for cls, info in counts.items():
            avg = sum(info["confidences"]) / len(info["confidences"])
            lines.append(f"  {cls:<25s}  count: {info['count']}   avg confidence: {avg:.1%}")
        return "\n".join(lines)

    def _draw_boxes(self, image_source, detections: list[dict]) -> np.ndarray:
        if isinstance(image_source, np.ndarray):
            img = image_source.copy()
        else:
            img = cv2.imread(str(image_source))
            if img is None:
                raise ValueError(f"Could not read image: {image_source}")

        unique_classes = list({d["class"] for d in detections})

        for det in detections:
            cls = det["class"]
            conf = det["confidence"]
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]

            color = BBOX_COLORS[unique_classes.index(cls) % len(BBOX_COLORS)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            label = f"{cls} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                img,
                label,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return img
