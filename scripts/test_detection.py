"""
ADMA -- Test Script

Run a trained model on any image and see detection results.

Usage:
  python scripts/test_detection.py --image photo.jpg --model military_vehicles_m
  python scripts/test_detection.py --list-models
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2

from adma.config import DEFAULT_CONFIDENCE, MODELS_DIR
from adma.config import OUTPUT_DIR as _OUTPUT_DIR
from adma.detector import MilitaryAssetDetector, discover_models

OUTPUT_DIR = Path(_OUTPUT_DIR)


def resolve_model(model_arg: str) -> Path:
    available = discover_models()
    if model_arg in available:
        return available[model_arg]

    path = Path(model_arg)
    if path.exists():
        return path

    print(f"[ERROR] Model '{model_arg}' not found.")
    if available:
        print("\nAvailable models:")
        for name, pt in available.items():
            print(f"  {name:<30s} {pt}")
    else:
        print(f"\nNo trained models found in {MODELS_DIR}/")
        print("Train one first:  python scripts/train.py --dataset <name>")
    sys.exit(1)


def list_models() -> None:
    available = discover_models()
    if not available:
        print(f"\nNo trained models found in {MODELS_DIR}/")
        print("Train one first:  python scripts/train.py --dataset <name>")
        return

    print(f"\nTrained models ({len(available)}):\n")
    for name, pt in available.items():
        size_mb = pt.stat().st_size / (1024 * 1024)
        print(f"  {name:<30s} {size_mb:>6.1f} MB   {pt}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Test military asset detection")
    parser.add_argument("--image", help="Path to input image")
    parser.add_argument("--model", help="Model name or path to .pt file")
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--output", default=str(OUTPUT_DIR))
    parser.add_argument("--list-models", action="store_true")
    args = parser.parse_args()

    if args.list_models:
        list_models()
        return

    if not args.image:
        parser.error("--image is required (or use --list-models)")
    if not args.model:
        available = discover_models()
        if len(available) == 1:
            args.model = next(iter(available))
            print(f"[INFO] Auto-selected model: {args.model}")
        elif available:
            print("[ERROR] Multiple models available. Specify one with --model:")
            for name in available:
                print(f"  {name}")
            sys.exit(1)
        else:
            print("[ERROR] No models found. Train one first:")
            print("  python scripts/train.py --dataset <name>")
            sys.exit(1)

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"[ERROR] Image not found: {image_path.resolve()}")
        sys.exit(1)

    model_path = resolve_model(args.model)
    detector = MilitaryAssetDetector(str(model_path), args.confidence)
    results = detector.detect(str(image_path), args.confidence)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"detected_{image_path.name}"
    cv2.imwrite(str(out_path), results["annotated_image"])

    print()
    print("=" * 60)
    print("  MILITARY ASSET DETECTION RESULTS")
    print("=" * 60)
    print(f"  Image      : {image_path.name}")
    print(f"  Model      : {args.model}")
    print(f"  Confidence : >= {args.confidence:.0%}")
    print("-" * 60)
    print()
    print(results["summary"])
    print()

    if results["detections"]:
        print("-" * 60)
        print("  Detailed detections:")
        print("-" * 60)
        for i, det in enumerate(results["detections"], 1):
            x1, y1, x2, y2 = det["bbox"]
            print(
                f"  {i:>2}. {det['class']:<25s} {det['confidence']:>6.1%}"
                f"   bbox: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]"
            )
        print()

    print(f"  Annotated image saved -> {out_path.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
