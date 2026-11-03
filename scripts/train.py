"""
ADMA -- Training Script

Fine-tunes a YOLOv8 model on a military asset dataset so the model
learns ONLY military classes and will never predict civilian objects.

Usage:
  python scripts/train.py --dataset military_vehicles --epochs 150 --model-size m
  python scripts/train.py --dataset all --epochs 150
  python scripts/train.py --data path/to/custom/data.yaml --name my_model
  python scripts/train.py --dataset all --seed 42  (seed defaults to 42)
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

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
from adma.datasets import (
    COMBINED_DATASET_EXCLUSIONS,
    DATASETS,
    LABEL_MAP,
)

COMBINED_DIR = Path(COMBINED_DATA_DIR)
DEFAULT_TRAIN_SEED = 42


def _io_path(path: Path) -> str:
    path_str = str(path)
    if sys.platform != "win32":
        return path_str

    absolute_path = os.path.abspath(path_str)
    if len(absolute_path) < 240:
        return path_str
    if absolute_path.startswith("\\\\"):
        return "\\\\?\\UNC\\" + absolute_path[2:]
    return "\\\\?\\" + absolute_path


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

    # Roboflow YAMLs can contain an absolute path from the machine that
    # downloaded them; the registry location is the local source of truth.
    base = loc.resolve()
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


def _map_label(raw_name: str, dataset_name: str) -> str | None:
    """Return the canonical class or None; reject unconfigured labels."""
    if raw_name not in LABEL_MAP:
        raise ValueError(
            f"Source label '{raw_name}' in dataset '{dataset_name}' "
            "is missing from LABEL_MAP. Add an explicit mapping or "
            "set the label to None."
        )

    return LABEL_MAP[raw_name]


def _build_unified_class_list(configs: list[dict]) -> list[str]:
    all_names: set[str] = set()

    for cfg in configs:
        for raw in cfg["names"]:
            mapped = _map_label(raw, cfg["name"])
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
        mapped = _map_label(old_name, cfg["name"])

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

        for image_index, img in enumerate(sorted(src_imgs.iterdir())):
            if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                output_stem = f"{image_index:06d}"
                dst = dst_imgs / f"{output_stem}{img.suffix.lower()}"
                shutil.copy2(_io_path(img), str(dst))

                lbl_src = src_lbls / (img.stem + ".txt")
                lbl_dst = dst_lbls / f"{output_stem}.txt"
                if os.path.isfile(_io_path(lbl_src)):
                    _rewrite_label_file(lbl_src, lbl_dst, old_to_new)

    train_out = output_base / "images" / "train" / ds_name
    val_out = output_base / "images" / "val" / ds_name
    return train_out, val_out


def _rewrite_label_file(src: Path, dst: Path, mapping: dict[int, int | None]) -> None:
    with open(_io_path(src), encoding="utf-8") as label_file:
        lines = label_file.read().strip().splitlines()
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
    with open(dst, "w", encoding="utf-8") as label_file:
        label_file.write("\n".join(new_lines) + "\n" if new_lines else "")


def _merge_all_datasets() -> Path:
    print("\n  Merging all downloaded datasets with label normalization ...\n")

    configs = []

    for name, info in DATASETS.items():
        if name in COMBINED_DATASET_EXCLUSIONS:
            print(f"  [SKIP] '{name}' excluded from combined training.")
            print(f"         {COMBINED_DATASET_EXCLUSIONS[name]}")
            continue

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


def _git_commit() -> str:
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def train(
    data_yaml: Path,
    output_name: str,
    epochs: int = TRAIN_EPOCHS,
    batch: int = TRAIN_BATCH_SIZE,
    imgsz: int = TRAIN_IMAGE_SIZE,
    model_size: str = DEFAULT_MODEL_SIZE,
    workers: int = TRAIN_WORKERS,
    seed: int = DEFAULT_TRAIN_SEED,
) -> Path:
    import mlflow

    resolved_data_yaml = data_yaml.resolve()
    git_commit = _git_commit()

    print("=" * 60)
    print("  MILITARY ASSET DETECTION -- TRAINING")
    print("=" * 60)
    print(f"  Model       : YOLOv8{model_size}")
    print(f"  Dataset     : {resolved_data_yaml}")
    print(f"  Output name : {output_name}")
    print(f"  Epochs      : {epochs}")
    print(f"  Batch size  : {batch}")
    print(f"  Image size  : {imgsz}")
    print(f"  Workers     : {workers}")
    print(f"  Seed        : {seed}")
    print("=" * 60)

    base_model = f"yolov8{model_size}.pt"
    model = YOLO(base_model)

    def log_training_metadata(trainer) -> None:
        if mlflow.active_run() is None:
            raise RuntimeError("Ultralytics did not start an active MLflow run")
        mlflow.log_params(
            {
                "data_yaml": str(resolved_data_yaml),
                "seed": seed,
                "git_commit": git_commit,
            }
        )

    model.add_callback("on_train_start", log_training_metadata)
    previous_keep_run_active = os.environ.get("MLFLOW_KEEP_RUN_ACTIVE")
    os.environ["MLFLOW_KEEP_RUN_ACTIVE"] = "True"

    try:
        model.train(
            data=str(resolved_data_yaml),
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
            seed=seed,
        )

        run_best = _find_best_pt(RUNS_DIR, output_name)

        models_dir = Path(MODELS_DIR) / output_name
        models_dir.mkdir(parents=True, exist_ok=True)
        dest = models_dir / "best.pt"
        shutil.copy2(str(run_best), str(dest))

        evaluation_dir = Path(RUNS_DIR) / "eval" / output_name
        evaluator_script = Path(__file__).resolve().with_name("evaluator.py")
        subprocess.run(
            [
                sys.executable,
                str(evaluator_script),
                "--data",
                str(resolved_data_yaml),
                "--model",
                str(run_best.resolve()),
                "--output-dir",
                str(evaluation_dir),
            ],
            cwd=Path(__file__).resolve().parent.parent,
            check=True,
        )
        metrics_path = evaluation_dir / "metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"Evaluator did not create the expected report: {metrics_path}")
        mlflow.log_artifact(str(metrics_path), artifact_path="evaluation")
    finally:
        try:
            mlflow.end_run()
        finally:
            if previous_keep_run_active is None:
                os.environ.pop("MLFLOW_KEEP_RUN_ACTIVE", None)
            else:
                os.environ["MLFLOW_KEEP_RUN_ACTIVE"] = previous_keep_run_active

    print()
    print("=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Model saved to    : {dest.resolve()}")
    print(f"  Training plots at : {run_best.parent.parent.resolve()}")
    print(f"  Evaluation report : {metrics_path.resolve()}")
    print()
    print("  Next steps:")
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
        "--seed",
        type=int,
        default=DEFAULT_TRAIN_SEED,
        help=f"Random seed for reproducible training (default: {DEFAULT_TRAIN_SEED})",
    )
    parser.add_argument(
        "--model-size",
        default=DEFAULT_MODEL_SIZE,
        choices=["n", "s", "m", "l", "x"],
        help="YOLOv8 size: n(ano) s(mall) m(edium) l(arge) x(tra-large)",
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

    train(
        yaml_path,
        out_name,
        args.epochs,
        args.batch,
        args.imgsz,
        args.model_size,
        args.workers,
        args.seed,
    )
