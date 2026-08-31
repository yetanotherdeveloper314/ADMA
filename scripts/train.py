"""
ADMA -- Training Script

Fine-tunes a YOLOv8 model on a military asset dataset so the model
learns ONLY military classes and will never predict civilian objects.

Usage:
  python scripts/train.py --dataset military_vehicles --epochs 150 --model-size m
  python scripts/train.py --dataset all --epochs 150
  python scripts/train.py --data path/to/custom/data.yaml --name my_model
"""

import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO

from adma.config import (
    COMBINED_DATA_DIR,
    DEFAULT_MODEL_SIZE,
    MODELS_DIR,
    RUNS_DIR,
    TRAIN_BATCH_SIZE,
    TRAIN_EPOCHS,
    TRAIN_IMAGE_SIZE,
    TRAIN_PATIENCE,
    TRAIN_WORKERS,
)
from adma.datasets import DATASETS, LABEL_MAP

COMBINED_DIR = Path(COMBINED_DATA_DIR)


def resolve_data_yaml(dataset_name: str) -> Path:
    if dataset_name == "all":
        return _merge_all_datasets()

    if dataset_name not in DATASETS:
        available = ", ".join(DATASETS.keys())
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {available}")

    yaml_path = Path(DATASETS[dataset_name]["location"]) / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Dataset '{dataset_name}' not downloaded yet.\n"
            f"Run:  python scripts/download_dataset.py --dataset {dataset_name}"
        )
    return yaml_path


# ------------------------------------------------------------------
# Combined-dataset merging with label normalization
# ------------------------------------------------------------------


def _load_dataset_config(name: str, info: dict) -> dict | None:
    loc = Path(info["location"])
    yaml_path = loc / "data.yaml"
    if not yaml_path.exists():
        print(f"  [SKIP] '{name}' not downloaded")
        return None

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    base = Path(cfg.get("path", str(loc.resolve())))
    train_dir = base / cfg["train"]
    val_dir = base / cfg["val"]

    names = cfg.get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names.keys())]

    return {
        "name": name,
        "base": base,
        "train_images": train_dir,
        "val_images": val_dir,
        "train_labels": Path(str(train_dir).replace("images", "labels")),
        "val_labels": Path(str(val_dir).replace("images", "labels")),
        "names": names,
    }


def _build_unified_class_list(configs: list[dict]) -> list[str]:
    all_names: set[str] = set()
    for cfg in configs:
        for raw in cfg["names"]:
            mapped = LABEL_MAP.get(raw, raw)
            if mapped is not None:
                all_names.add(mapped)
    return sorted(all_names)


def _remap_labels_for_dataset(
    cfg: dict,
    unified_names: list[str],
    output_base: Path,
) -> tuple[Path, Path]:
    """Copy images and rewrite label files with unified, normalized indices."""
    old_to_new: dict[int, int | None] = {}
    for old_idx, old_name in enumerate(cfg["names"]):
        mapped = LABEL_MAP.get(old_name, old_name)
        if mapped is None:
            old_to_new[old_idx] = None
        else:
            old_to_new[old_idx] = unified_names.index(mapped)

    ds_name = cfg["name"]

    for split in ("train", "val"):
        src_imgs = cfg[f"{split}_images"]
        src_lbls = cfg[f"{split}_labels"]
        dst_imgs = output_base / "images" / split / ds_name
        dst_lbls = output_base / "labels" / split / ds_name
        dst_imgs.mkdir(parents=True, exist_ok=True)
        dst_lbls.mkdir(parents=True, exist_ok=True)

        if not src_imgs.exists():
            print(f"    [WARN] {split} images missing for '{ds_name}', skipping")
            continue

        for img in src_imgs.iterdir():
            if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                dst = dst_imgs / img.name
                if not dst.exists():
                    shutil.copy2(str(img), str(dst))

                lbl_src = src_lbls / (img.stem + ".txt")
                lbl_dst = dst_lbls / (img.stem + ".txt")
                if lbl_src.exists() and not lbl_dst.exists():
                    _rewrite_label_file(lbl_src, lbl_dst, old_to_new)

    train_out = output_base / "images" / "train" / ds_name
    val_out = output_base / "images" / "val" / ds_name
    return train_out, val_out


def _rewrite_label_file(src: Path, dst: Path, mapping: dict[int, int | None]) -> None:
    lines = src.read_text().strip().splitlines()
    new_lines = []
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        old_cls = int(parts[0])
        new_cls = mapping.get(old_cls)
        if new_cls is None:
            continue
        parts[0] = str(new_cls)
        new_lines.append(" ".join(parts))
    dst.write_text("\n".join(new_lines) + "\n" if new_lines else "")


def _merge_all_datasets() -> Path:
    print("\n  Merging all downloaded datasets with label normalization ...\n")

    configs = []
    for name, info in DATASETS.items():
        cfg = _load_dataset_config(name, info)
        if cfg is not None:
            configs.append(cfg)

    if not configs:
        raise FileNotFoundError(
            "No datasets downloaded. Run:  python scripts/download_dataset.py --dataset all"
        )

    unified_names = _build_unified_class_list(configs)
    print(f"  Unified class list ({len(unified_names)} classes):")
    for i, n in enumerate(unified_names):
        print(f"    {i}: {n}")
    print()

    COMBINED_DIR.mkdir(parents=True, exist_ok=True)

    all_train_dirs, all_val_dirs = [], []
    for cfg in configs:
        print(f"  Remapping '{cfg['name']}' ({len(cfg['names'])} raw -> normalized) ...")
        train_dir, val_dir = _remap_labels_for_dataset(cfg, unified_names, COMBINED_DIR)
        all_train_dirs.append(str(train_dir))
        all_val_dirs.append(str(val_dir))

    combined_abs = COMBINED_DIR.resolve()
    combined_yaml = {
        "path": str(combined_abs),
        "train": [str(Path(d).resolve().relative_to(combined_abs)) for d in all_train_dirs],
        "val": [str(Path(d).resolve().relative_to(combined_abs)) for d in all_val_dirs],
        "nc": len(unified_names),
        "names": unified_names,
    }

    out = COMBINED_DIR / "data.yaml"
    with open(out, "w") as f:
        yaml.dump(combined_yaml, f, default_flow_style=False, sort_keys=False)

    total_images = sum(
        len(list(Path(d).glob("*"))) for d in all_train_dirs + all_val_dirs if Path(d).exists()
    )
    print(f"\n  [OK] Combined {len(configs)} dataset(s), {total_images} total images")
    print(f"  [OK] Config written to {out.resolve()}\n")

    for cache in COMBINED_DIR.rglob("*.cache"):
        cache.unlink()

    return out


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------


def model_output_name(dataset_name: str, model_size: str, custom_name: str | None) -> str:
    if custom_name:
        return custom_name
    return f"{dataset_name}_{model_size}"


def train(
    data_yaml: Path,
    output_name: str,
    epochs: int = TRAIN_EPOCHS,
    batch: int = TRAIN_BATCH_SIZE,
    imgsz: int = TRAIN_IMAGE_SIZE,
    model_size: str = DEFAULT_MODEL_SIZE,
    workers: int = TRAIN_WORKERS,
) -> Path:
    print("=" * 60)
    print("  MILITARY ASSET DETECTION -- TRAINING")
    print("=" * 60)
    print(f"  Model       : YOLO11{model_size}")
    print(f"  Dataset     : {data_yaml.resolve()}")
    print(f"  Output name : {output_name}")
    print(f"  Epochs      : {epochs}")
    print(f"  Batch size  : {batch}")
    print(f"  Image size  : {imgsz}")
    print(f"  Workers     : {workers}")
    print("=" * 60)

    base_model = f"yolo11{model_size}.pt"
    model = YOLO(base_model)

    model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        project=RUNS_DIR,
        name=output_name,
        exist_ok=True,
        patience=TRAIN_PATIENCE,
        save=True,
        plots=True,
        verbose=True,
        workers=workers,
    )

    run_best = _find_best_pt(RUNS_DIR, output_name)

    models_dir = Path(MODELS_DIR) / output_name
    models_dir.mkdir(parents=True, exist_ok=True)
    dest = models_dir / "best.pt"
    shutil.copy2(str(run_best), str(dest))

    print()
    print("=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Model saved to    : {dest.resolve()}")
    print(f"  Training plots at : {run_best.parent.parent.resolve()}")

    print()
    print("  Next steps:")
    print(
        f"    python scripts/evaluator.py --data {data_yaml} --gt-images ... --gt-labels ... --pred-labels ..."
    )
    print(f"    python scripts/test_detection.py --image YOUR_IMAGE.jpg --model {output_name}")
    print(f"    python -m adma.app   (select '{output_name}' from the dropdown)")
    print("=" * 60)

    return dest


def _find_best_pt(project: str, name: str) -> Path:
    candidates = [
        Path(project) / name / "weights" / "best.pt",
        Path(project) / "detect" / name / "weights" / "best.pt",
        Path(project) / "detect" / project / name / "weights" / "best.pt",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Could not find best.pt under {project}/{name}. Checked: {[str(c) for c in candidates]}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train military asset detector")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", help="Dataset name from registry (or 'all')")
    group.add_argument("--data", help="Path to a custom data.yaml file")

    parser.add_argument("--name", help="Custom model output name")
    parser.add_argument("--epochs", type=int, default=TRAIN_EPOCHS)
    parser.add_argument("--batch", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--imgsz", type=int, default=TRAIN_IMAGE_SIZE)
    parser.add_argument("--workers", type=int, default=TRAIN_WORKERS)
    parser.add_argument(
        "--model-size",
        default=DEFAULT_MODEL_SIZE,
        choices=["n", "s", "m", "l", "x"],
        help="YOLO11 size: n(ano) s(mall) m(edium) l(arge) x(tra-large)",
    )
    args = parser.parse_args()

    if args.dataset:
        yaml_path = resolve_data_yaml(args.dataset)
        out_name = model_output_name(args.dataset, args.model_size, args.name)
    else:
        yaml_path = Path(args.data)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Data YAML not found: {yaml_path}")
        out_name = args.name or yaml_path.stem

    train(yaml_path, out_name, args.epochs, args.batch, args.imgsz, args.model_size, args.workers)
