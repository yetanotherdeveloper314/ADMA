"""
ADMA -- Evaluation Script

Scores any detection model on a YOLO-format dataset split: COCO mAP50-95,
mAP50, mAP75, per-class precision/recall/F1 at an operating confidence,
a confusion matrix, and per-image latency when the model is run live.

Predictions come from exactly one source:
  --model        trained model name (from models/) or Ultralytics weights path
  --pred-labels  directory of YOLO txt predictions with a confidence column
  --pred-json    JSON detection records from any other model

Usage:
  python scripts/evaluator.py --dataset all --model all_m
  python scripts/evaluator.py --data data/combined/data.yaml --model path/to/best.pt
  python scripts/evaluator.py --data data/combined/data.yaml --pred-labels runs/detect/predict/labels
  python scripts/evaluator.py --data data/combined/data.yaml --pred-json preds.json --class-map map.yaml
"""

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml

from adma.config import (
    COMBINED_DATA_DIR,
    DEFAULT_CONFIDENCE,
    MODELS_DIR,
    RUNS_DIR,
    TRAIN_IMAGE_SIZE,
)
from adma.datasets import DATASETS
from adma.evaluator import (
    DEFAULT_MAX_DET,
    UltralyticsPredictor,
    evaluate,
    format_report,
    load_dataset,
    load_json_predictions,
    load_yolo_predictions,
    predict_with,
    save_report,
)


def resolve_data_yaml(dataset_name: str) -> Path:
    if dataset_name == "all":
        yaml_path = Path(COMBINED_DATA_DIR) / "data.yaml"
        hint = "python scripts/train.py --dataset all   (builds the combined dataset)"
    elif dataset_name in DATASETS:
        yaml_path = Path(DATASETS[dataset_name]["location"]) / "data.yaml"
        hint = f"python scripts/download_dataset.py --dataset {dataset_name}"
    else:
        available = ", ".join(["all", *DATASETS])
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {available}")

    if not yaml_path.exists():
        raise FileNotFoundError(f"{yaml_path} not found. Run:  {hint}")
    return yaml_path


def resolve_model(model_arg: str) -> Path:
    named = Path(MODELS_DIR) / model_arg / "best.pt"
    if named.exists():
        return named
    path = Path(model_arg)
    if path.exists():
        return path
    raise FileNotFoundError(
        f"Model '{model_arg}' not found as {named} or as a file path.\n"
        "List trained models with:  python scripts/test_detection.py --list-models"
    )


def load_class_map(path: str | None) -> dict[str, str | None] | None:
    if path is None:
        return None
    with open(path) as f:
        mapping = yaml.safe_load(f) or {}
    if not isinstance(mapping, dict):
        raise ValueError(f"Class map {path} must map model class names to dataset class names")
    return {str(k): (None if v is None else str(v)) for k, v in mapping.items()}


def print_progress(done: int, total: int) -> None:
    if done == total or done % 100 == 0:
        print(f"\r  Predicting   : {done}/{total} images", end="\n" if done == total else "")


def main(argv: list[str] | None = None) -> dict[str, Path]:
    parser = argparse.ArgumentParser(description="Evaluate a military asset detector")
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--dataset", help="Dataset name from registry (or 'all')")
    data_group.add_argument("--data", help="Path to a data.yaml file")
    parser.add_argument("--split", default="val", help="Split to evaluate (default: val)")

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", help="Model name in models/ or path to Ultralytics weights")
    source.add_argument("--pred-labels", help="Directory of YOLO txt predictions with confidence")
    source.add_argument("--pred-json", help="JSON file of pixel-space detection records")

    parser.add_argument(
        "--class-map",
        help="YAML/JSON mapping model class names to dataset class names (null drops a class)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help=f"Operating confidence for P/R/F1 and confusion matrix (default: {DEFAULT_CONFIDENCE})",
    )
    parser.add_argument(
        "--iou", type=float, default=0.5, help="IoU for P/R/F1 and confusion matrix (default: 0.5)"
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=DEFAULT_MAX_DET,
        help=f"Predictions kept per image (default: {DEFAULT_MAX_DET})",
    )
    parser.add_argument("--imgsz", type=int, default=TRAIN_IMAGE_SIZE, help="Inference size")
    parser.add_argument("--device", help="Inference device, e.g. cpu, 0, mps")
    parser.add_argument("--output-dir", help="Report directory (default: runs/eval/<source name>)")
    args = parser.parse_args(argv)

    data_yaml = resolve_data_yaml(args.dataset) if args.dataset else Path(args.data)
    class_map = load_class_map(args.class_map)

    if args.model:
        model_path = resolve_model(args.model)
        source_desc = str(model_path.resolve())
        parent = model_path.parent
        run_name = parent.parent.name if parent.name == "weights" else parent.name
    else:
        source_path = Path(args.pred_labels or args.pred_json)
        source_desc = str(source_path.resolve())
        run_name = source_path.stem if args.pred_json else source_path.name
    output_dir = Path(args.output_dir) if args.output_dir else Path(RUNS_DIR) / "eval" / run_name

    print("=" * 60)
    print("  MILITARY ASSET DETECTION -- EVALUATION")
    print("=" * 60)
    print(f"  Predictions  : {source_desc}")
    print(f"  Dataset      : {data_yaml.resolve()}")
    print(f"  Split        : {args.split}")
    print("=" * 60)

    dataset = load_dataset(data_yaml, args.split)
    print(f"  Images       : {len(dataset.images)}")
    print(f"  GT boxes     : {len(dataset.ground_truth)}")

    latency = None
    if args.model:
        predictor = UltralyticsPredictor(
            model_path, imgsz=args.imgsz, max_det=args.max_det, device=args.device
        )
        predictions, latency = predict_with(predictor, dataset, class_map, progress=print_progress)
    elif args.pred_labels:
        predictions = load_yolo_predictions(args.pred_labels, dataset)
    else:
        predictions = load_json_predictions(args.pred_json, dataset, class_map)
    print(f"  Pred boxes   : {len(predictions)}")
    print()

    report = evaluate(dataset, predictions, args.conf, args.iou, args.max_det, latency)
    print(format_report(report))

    metadata = {
        "predictions": source_desc,
        "data": str(data_yaml.resolve()),
        "split": args.split,
        "class_map": class_map,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    paths = save_report(report, output_dir, metadata)

    print()
    print("=" * 60)
    print(f"  Reports written to {output_dir.resolve()}")
    print("=" * 60)
    return paths


if __name__ == "__main__":
    main()
