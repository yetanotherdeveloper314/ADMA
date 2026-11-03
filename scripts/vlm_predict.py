"""
ADMA -- Vision-language model predictions

Asks a local Qwen2.5-VL model (through Ollama) for the boxes in each image of a
split and writes them as JSON detection records that scripts/evaluator.py reads
with --pred-json. Qwen gives no confidence scores, so every box gets 1.0: judge
it on precision, recall, F1 and counts, not mAP.

Next to the predictions it writes <out>.latency.json (seconds per image) and
<out>.replies.jsonl (every raw reply, for checking the ones that failed).

Usage:
  python scripts/vlm_predict.py --data data/combined_v1/data.yaml --out runs/qwen/qwen_10.json --limit 10 --draw 10
  python scripts/vlm_predict.py --data data/combined_v1/data.yaml --out runs/qwen/qwen_300.json --limit 300
  python scripts/evaluator.py --data data/combined_v1/data.yaml --pred-json runs/qwen/qwen_300.json
"""

import argparse
import json
import math
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from PIL import Image, ImageDraw, ImageOps

from adma.evaluator import load_dataset

DEFAULT_MODEL = "qwen2.5vl:7b"
# Qwen2.5-VL resizes images to multiples of 28 pixels before reading them.
RESIZE_FACTOR = 28
MIN_PIXELS = 56 * 56
DEFAULT_MAX_PIXELS = 28 * 28 * 1280

PROMPT = (
    "Find every military vehicle in this image. Use only these labels: {labels}. "
    'Reply with JSON only: a list like [{{"bbox_2d": [x1, y1, x2, y2], "label": "tank"}}] '
    "in pixel coordinates. Reply [] if there are none."
)


def parse_reply(reply: str, classes: list[str]) -> list[dict] | None:
    """Boxes from a model reply, or None when the reply holds no usable JSON list."""
    match = re.search(r"\[.*\]", reply, re.DOTALL)
    if match is None:
        return None
    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(items, list):
        return None
    boxes = []
    for item in items:
        if not isinstance(item, dict) or item.get("label") not in classes:
            continue
        bbox = item.get("bbox_2d")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            boxes.append({"class": item["label"], "bbox": [float(v) for v in bbox]})
        except (TypeError, ValueError):
            continue
    return boxes


def smart_resize(width: int, height: int, max_pixels: int = DEFAULT_MAX_PIXELS) -> tuple[int, int]:
    """(width, height) of the copy Qwen2.5-VL actually sees, following its own resize rule."""
    w_bar = max(RESIZE_FACTOR, round(width / RESIZE_FACTOR) * RESIZE_FACTOR)
    h_bar = max(RESIZE_FACTOR, round(height / RESIZE_FACTOR) * RESIZE_FACTOR)
    if w_bar * h_bar > max_pixels:
        beta = math.sqrt(width * height / max_pixels)
        w_bar = math.floor(width / beta / RESIZE_FACTOR) * RESIZE_FACTOR
        h_bar = math.floor(height / beta / RESIZE_FACTOR) * RESIZE_FACTOR
    elif w_bar * h_bar < MIN_PIXELS:
        beta = math.sqrt(MIN_PIXELS / (width * height))
        w_bar = math.ceil(width * beta / RESIZE_FACTOR) * RESIZE_FACTOR
        h_bar = math.ceil(height * beta / RESIZE_FACTOR) * RESIZE_FACTOR
    return w_bar, h_bar


def rescale_bbox(
    bbox: list[float], from_size: tuple[int, int], to_size: tuple[int, int]
) -> list[float]:
    sx = to_size[0] / from_size[0]
    sy = to_size[1] / from_size[1]
    x1, y1, x2, y2 = bbox
    return [x1 * sx, y1 * sy, x2 * sx, y2 * sy]


def draw_check(path: Path, gt_boxes: list, pred_boxes: list[dict], out_path: Path) -> None:
    """True boxes in green, model boxes in red, so shifted or shrunk boxes stand out."""
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    width, height = image.size
    draw = ImageDraw.Draw(image)
    for x1, y1, x2, y2 in gt_boxes:
        draw.rectangle([x1 * width, y1 * height, x2 * width, y2 * height], outline="lime", width=3)
    for box in pred_boxes:
        draw.rectangle(box["bbox"], outline="red", width=3)
        draw.text((box["bbox"][0] + 3, box["bbox"][1] + 3), box["class"], fill="red")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Get Qwen2.5-VL boxes for a dataset split.")
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--split", default="val", help="Split to predict (default: val)")
    parser.add_argument("--out", required=True, help="Predictions JSON to write")
    parser.add_argument("--limit", type=int, default=300, help="First N images, sorted by id")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help=f"Ollama model (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--rescale-from-resized",
        action="store_true",
        help="Treat reply coordinates as pixels of Qwen's resized copy and map them back",
    )
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    parser.add_argument("--draw", type=int, default=0, help="Draw boxes for the first N images")
    args = parser.parse_args()

    import ollama

    dataset = load_dataset(args.data, args.split)
    classes = list(dataset.class_names.values())
    prompt = PROMPT.format(labels=", ".join(classes))
    gt_by_image: dict[str, list] = {}
    for gt in dataset.ground_truth:
        gt_by_image.setdefault(gt.image_id, []).append(gt.bbox)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    draw_dir = out_path.parent / "check"
    images = sorted(dataset.images.items())[: args.limit]

    records: list[dict] = []
    seconds: list[float] = []
    bad = 0
    with open(out_path.with_suffix(".replies.jsonl"), "w", encoding="utf-8") as replies_file:
        for index, (image_id, path) in enumerate(images):
            start = time.perf_counter()
            reply = ollama.chat(
                model=args.model,
                messages=[{"role": "user", "content": prompt, "images": [str(path)]}],
                options={"temperature": 0},
            )["message"]["content"]
            seconds.append(time.perf_counter() - start)
            replies_file.write(json.dumps({"image": image_id, "reply": reply}) + "\n")

            boxes = parse_reply(reply, classes)
            if boxes is None:
                bad += 1
                continue
            if args.rescale_from_resized:
                with Image.open(path) as opened:
                    size = ImageOps.exif_transpose(opened).size
                resized = smart_resize(*size, max_pixels=args.max_pixels)
                for box in boxes:
                    box["bbox"] = rescale_bbox(box["bbox"], resized, size)
            for box in boxes:
                records.append({"image": image_id, "confidence": 1.0, **box})
            if index < args.draw:
                name = image_id.replace("/", "__") + ".jpg"
                draw_check(path, gt_by_image.get(image_id, []), boxes, draw_dir / name)
            print(f"  [{index + 1}/{len(images)}] {image_id}: {len(boxes)} boxes")

    out_path.write_text(json.dumps(records))
    latency = {
        "images": len(seconds),
        "mean_s": statistics.mean(seconds),
        "median_s": statistics.median(seconds),
        "p95_s": statistics.quantiles(seconds, n=20)[-1] if len(seconds) > 1 else seconds[0],
    }
    out_path.with_suffix(".latency.json").write_text(json.dumps(latency, indent=2))

    print()
    print(f"  Boxes              : {len(records)}")
    print(f"  Unparseable replies: {bad} of {len(images)} ({bad / len(images):.1%})")
    print(f"  Seconds per image  : mean {latency['mean_s']:.1f}, median {latency['median_s']:.1f}")
    print(f"  Predictions        : {out_path}")
    if args.draw:
        print(f"  Box check images   : {draw_dir}")


if __name__ == "__main__":
    main()
