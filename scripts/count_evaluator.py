"""Compare per-image object counts against YOLO ground truth."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from adma.evaluator import load_dataset, load_json_predictions


def main():
    parser = argparse.ArgumentParser(description="Evaluate per-image, per-class object counts.")
    parser.add_argument("--data", required=True, help="YOLO data.yaml")
    parser.add_argument("--split", default="val")
    parser.add_argument("--pred-json", required=True)
    parser.add_argument("--min-confidence", type=float, default=0.25)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    if not 0.0 <= args.min_confidence <= 1.0:
        parser.error("--min-confidence must be between 0 and 1")
    if args.limit < 1:
        parser.error("--limit must be at least 1")

    dataset = load_dataset(args.data, split=args.split)
    selected_items = sorted(dataset.images.items())[: args.limit]
    selected_ids = {image_id for image_id, _ in selected_items}

    if not selected_items:
        raise SystemExit("No images found in the selected dataset split.")

    # Load predictions using the evaluator's existing JSON parser.
    predictions = load_json_predictions(args.pred_json, dataset)

    # Count only predictions belonging to the selected images and
    # meeting the requested confidence threshold.
    pred_counts = Counter(
        (p.image_id, p.class_id)
        for p in predictions
        if p.image_id in selected_ids and p.score >= args.min_confidence
    )

    # Count ground-truth objects for the same selected images.
    gt_counts = Counter(
        (gt.image_id, gt.class_id) for gt in dataset.ground_truth if gt.image_id in selected_ids
    )

    class_ids = sorted(dataset.class_names)
    image_ids = [image_id for image_id, _ in selected_items]

    output_dir = args.out_dir or (Path(args.pred_json).parent / "count_report")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    class_errors = Counter()
    class_exact = Counter()

    total_absolute_error = 0
    total_exact_pairs = 0
    exact_images = 0

    for image_id in image_ids:
        image_is_exact = True

        for class_id in class_ids:
            gt_count = gt_counts[(image_id, class_id)]
            pred_count = pred_counts[(image_id, class_id)]
            error = abs(pred_count - gt_count)
            exact = pred_count == gt_count

            class_errors[class_id] += error
            class_exact[class_id] += int(exact)
            total_absolute_error += error
            total_exact_pairs += int(exact)

            if not exact:
                image_is_exact = False

            rows.append(
                {
                    "image": image_id,
                    "class": dataset.class_names[class_id],
                    "ground_truth": gt_count,
                    "predicted": pred_count,
                    "absolute_error": error,
                    "exact": exact,
                }
            )

        exact_images += int(image_is_exact)

    number_of_images = len(image_ids)
    number_of_classes = len(class_ids)
    total_pairs = number_of_images * number_of_classes

    class_summary = []
    for class_id in class_ids:
        class_summary.append(
            {
                "class": dataset.class_names[class_id],
                "mae": class_errors[class_id] / number_of_images,
                "exact_count_share": class_exact[class_id] / number_of_images,
            }
        )

    summary = {
        "prediction_file": str(Path(args.pred_json)),
        "split": args.split,
        "images": number_of_images,
        "classes": number_of_classes,
        "image_class_pairs": total_pairs,
        "minimum_confidence": args.min_confidence,
        "overall_count_mae": total_absolute_error / total_pairs,
        "exact_image_class_share": total_exact_pairs / total_pairs,
        "exact_image_share": exact_images / number_of_images,
        "class_summary": class_summary,
    }

    with (output_dir / "per_image_class_counts.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with (output_dir / "class_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["class", "mae", "exact_count_share"])
        writer.writeheader()
        writer.writerows(class_summary)

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Images evaluated: {number_of_images}")
    print(f"Classes evaluated: {number_of_classes}")
    print(f"Image/class pairs: {total_pairs}")
    print(f"Overall count MAE: {summary['overall_count_mae']:.4f}")
    print(f"Exact image/class share: {summary['exact_image_class_share']:.2%}")
    print(f"Exact image share: {summary['exact_image_share']:.2%}")
    print("\nPer-class results:")
    for item in class_summary:
        print(f"  {item['class']}: MAE={item['mae']:.4f}, exact={item['exact_count_share']:.2%}")

    print(f"\nReports saved to: {output_dir}")


if __name__ == "__main__":
    main()
