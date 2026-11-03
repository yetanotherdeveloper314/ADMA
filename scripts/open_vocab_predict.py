"""
ADMA -- Open-vocabulary detector predictions

Runs a detector that takes class names as text (no training on our data) over a
split and writes JSON detection records that scripts/evaluator.py reads with
--pred-json. Unlike chat-style models these give confidence scores, so mAP,
precision/recall at any confidence and calibration all apply.

Frameworks:
  yolo-world  Ultralytics YOLO-World (default weights: yolov8s-worldv2.pt)
  owlv2       Hugging Face OWLv2 (default: google/owlv2-base-patch16-ensemble)

Each class is described by a text prompt: its name with underscores as spaces,
unless --prompts gives a YAML file mapping class names to other wording.

Next to the predictions it writes <out>.latency.json (ms per image, after one
untimed warm-up image).

Usage:
  python scripts/open_vocab_predict.py --framework yolo-world --data data/combined_v1/data.yaml --out runs/open_vocab/yolo_world.json
  python scripts/open_vocab_predict.py --framework owlv2 --data data/combined_v1/data.yaml --out runs/open_vocab/owlv2.json --limit 300
  python scripts/evaluator.py --data data/combined_v1/data.yaml --pred-json runs/open_vocab/yolo_world.json
"""

import argparse
import json
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml
from PIL import Image, ImageDraw, ImageOps

from adma.evaluator import load_dataset

DEFAULT_WEIGHTS = {
    "yolo-world": "yolov8s-worldv2.pt",
    "owlv2": "google/owlv2-base-patch16-ensemble",
}
# Keep low-confidence boxes: AP needs them. The evaluator keeps the top 100 per image.
DEFAULT_CONF = 0.001
OWLV2_THRESHOLD = 0.01

Predictor = Callable[[Path], list[dict]]


def build_prompts(class_names: list[str], overrides: dict[str, str] | None = None) -> list[str]:
    """One text prompt per class, in class order."""
    overrides = overrides or {}
    unknown = set(overrides) - set(class_names)
    if unknown:
        raise ValueError(
            f"--prompts names classes not in the dataset: {', '.join(sorted(unknown))}"
        )
    return [overrides.get(name, name.replace("_", " ")) for name in class_names]


def latency_stats(ms: list[float]) -> dict:
    return {
        "images": len(ms),
        "mean_ms": statistics.mean(ms),
        "median_ms": statistics.median(ms),
        "p95_ms": statistics.quantiles(ms, n=20)[-1] if len(ms) > 1 else ms[0],
    }


def owlv2_to_boxes(raw: list[dict], prompts: list[str], class_names: list[str]) -> list[dict]:
    """Convert Hugging Face zero-shot pipeline output to class/confidence/bbox records."""
    prompt_to_class = dict(zip(prompts, class_names, strict=True))
    boxes = []
    for item in raw:
        box = item["box"]
        boxes.append(
            {
                "class": prompt_to_class[item["label"]],
                "confidence": float(item["score"]),
                "bbox": [box["xmin"], box["ymin"], box["xmax"], box["ymax"]],
            }
        )
    return boxes


def yolo_world_predictor(
    weights: str, prompts: list[str], class_names: list[str], device
) -> Predictor:
    from ultralytics import YOLOWorld

    model = YOLOWorld(weights)
    model.set_classes(prompts)

    def predict(path: Path) -> list[dict]:
        result = model.predict(str(path), conf=DEFAULT_CONF, device=device, verbose=False)[0]
        boxes = result.boxes
        return [
            {"class": class_names[int(cls)], "confidence": float(conf), "bbox": xyxy}
            for xyxy, cls, conf in zip(
                boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist(), strict=True
            )
        ]

    return predict


def owlv2_predictor(weights: str, prompts: list[str], class_names: list[str], device) -> Predictor:
    from transformers import pipeline

    detector = pipeline("zero-shot-object-detection", model=weights, device=device)

    def predict(path: Path) -> list[dict]:
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        raw = detector(image, candidate_labels=prompts, threshold=OWLV2_THRESHOLD)
        return owlv2_to_boxes(raw, prompts, class_names)

    return predict


PREDICTORS = {"yolo-world": yolo_world_predictor, "owlv2": owlv2_predictor}


def draw_check(
    path: Path, gt_boxes: list, pred_boxes: list[dict], out_path: Path, min_conf: float
) -> None:
    """True boxes in green, model boxes at or above ``min_conf`` in red."""
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    width, height = image.size
    draw = ImageDraw.Draw(image)
    for x1, y1, x2, y2 in gt_boxes:
        draw.rectangle([x1 * width, y1 * height, x2 * width, y2 * height], outline="lime", width=3)
    for box in pred_boxes:
        if box["confidence"] < min_conf:
            continue
        draw.rectangle(box["bbox"], outline="red", width=3)
        label = f"{box['class']} {box['confidence']:.2f}"
        draw.text((box["bbox"][0] + 3, box["bbox"][1] + 3), label, fill="red")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Get open-vocabulary detector boxes for a split.")
    parser.add_argument("--framework", required=True, choices=sorted(PREDICTORS))
    parser.add_argument("--data", required=True, help="Path to a data.yaml file")
    parser.add_argument("--split", default="val", help="Split to predict (default: val)")
    parser.add_argument("--out", required=True, help="Predictions JSON to write")
    parser.add_argument("--weights", help="Weights file or Hugging Face model id")
    parser.add_argument("--prompts", help="YAML file mapping class names to prompt text")
    parser.add_argument("--limit", type=int, help="First N images, sorted by id (default: all)")
    parser.add_argument("--device", help="Inference device, e.g. cpu or 0 (default: GPU if found)")
    parser.add_argument("--draw", type=int, default=0, help="Draw boxes for the first N images")
    parser.add_argument(
        "--draw-conf", type=float, default=0.25, help="Lowest confidence drawn (default: 0.25)"
    )
    args = parser.parse_args()

    import torch

    device = args.device if args.device is not None else (0 if torch.cuda.is_available() else "cpu")
    dataset = load_dataset(args.data, args.split)
    class_names = [dataset.class_names[i] for i in sorted(dataset.class_names)]
    overrides = yaml.safe_load(Path(args.prompts).read_text()) if args.prompts else None
    prompts = build_prompts(class_names, overrides)
    weights = args.weights or DEFAULT_WEIGHTS[args.framework]
    predict = PREDICTORS[args.framework](weights, prompts, class_names, device)

    gt_by_image: dict[str, list] = {}
    for gt in dataset.ground_truth:
        gt_by_image.setdefault(gt.image_id, []).append(gt.bbox)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    draw_dir = out_path.parent / f"check_{out_path.stem}"
    images = sorted(dataset.images.items())[: args.limit]

    print(f"  {args.framework} ({weights}) on {len(images)} {args.split} images, device {device}")
    for name, prompt in zip(class_names, prompts, strict=True):
        print(f"    {name:<22} <- '{prompt}'")
    predict(images[0][1])  # warm-up, not timed

    records: list[dict] = []
    ms: list[float] = []
    for index, (image_id, path) in enumerate(images):
        start = time.perf_counter()
        boxes = predict(path)
        ms.append((time.perf_counter() - start) * 1000)
        records.extend({"image": image_id, **box} for box in boxes)
        if index < args.draw:
            name = image_id.replace("/", "__") + ".jpg"
            draw_check(path, gt_by_image.get(image_id, []), boxes, draw_dir / name, args.draw_conf)
        if (index + 1) % 100 == 0:
            print(f"  [{index + 1}/{len(images)}]")

    out_path.write_text(json.dumps(records))
    latency = latency_stats(ms)
    out_path.with_suffix(".latency.json").write_text(json.dumps(latency, indent=2))

    print()
    print(f"  Boxes        : {len(records)}")
    print(f"  ms per image : mean {latency['mean_ms']:.1f}, median {latency['median_ms']:.1f}")
    print(f"  Predictions  : {out_path}")
    if args.draw:
        print(f"  Box check    : {draw_dir}")


if __name__ == "__main__":
    main()
