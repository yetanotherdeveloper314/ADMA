"""
Download datasets from the registry.

Usage:
  python scripts/download_dataset.py --list
  python scripts/download_dataset.py --dataset military_vehicles
  python scripts/download_dataset.py --dataset all
"""

import argparse
import random
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml

from adma.config import RANDOM_SEED, ROBOFLOW_API_KEY, VAL_SPLIT_RATIO
from adma.datasets import DATASETS


def ensure_roboflow():
    try:
        from roboflow import Roboflow  # noqa: F401
    except ImportError:
        print("Installing roboflow package ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "roboflow"])


def download_one(name: str) -> Path:
    info = DATASETS[name]
    location = Path(info["location"])

    if (location / "data.yaml").exists():
        print(f"[SKIP] '{name}' already downloaded at {location.resolve()}")
        return location

    ensure_roboflow()
    from roboflow import Roboflow

    if not ROBOFLOW_API_KEY:
        print("[ERROR] ROBOFLOW_API_KEY is not set.")
        print("Add it to your .env file or export it in your shell.")
        print("See .env.example for reference.")
        sys.exit(1)

    print(f"\nDownloading '{name}' -- {info['description']} ...")
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(info["workspace"]).project(info["project"])
    project.version(info["version"]).download("yolov8", location=str(location))
    print(f"[OK] Saved to {location.resolve()}")

    _fix_dataset(location)
    return location


def _fix_dataset(location: Path) -> None:
    yaml_path = location / "data.yaml"
    if not yaml_path.exists():
        return

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    changed = False

    for key in ("train", "val", "test"):
        if key not in cfg:
            continue
        raw = cfg[key]
        if isinstance(raw, str) and raw.startswith(".."):
            fixed = raw.lstrip("./").lstrip("\\")
            cfg[key] = fixed
            changed = True

    if "path" not in cfg or not Path(cfg["path"]).is_absolute():
        cfg["path"] = str(location.resolve())
        changed = True

    if changed:
        with open(yaml_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        print(f"[FIX] Corrected paths in {yaml_path}")

    base = Path(cfg["path"])
    train_imgs = base / cfg.get("train", "train/images")
    val_imgs = base / cfg.get("val", "valid/images")

    if train_imgs.exists() and not val_imgs.exists():
        print("[FIX] No validation split found -- creating one (20% of training data) ...")
        _create_val_split(train_imgs, val_imgs)

    for cache in location.rglob("*.cache"):
        cache.unlink()


def _create_val_split(train_imgs: Path, val_imgs: Path) -> None:
    train_lbls = Path(str(train_imgs).replace("images", "labels"))
    val_lbls = Path(str(val_imgs).replace("images", "labels"))

    val_imgs.mkdir(parents=True, exist_ok=True)
    val_lbls.mkdir(parents=True, exist_ok=True)

    images = [
        f
        for f in train_imgs.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ]
    random.seed(RANDOM_SEED)
    random.shuffle(images)
    split_count = max(1, int(len(images) * VAL_SPLIT_RATIO))
    to_move = images[:split_count]

    for img in to_move:
        shutil.move(str(img), str(val_imgs / img.name))
        lbl = train_lbls / (img.stem + ".txt")
        if lbl.exists():
            shutil.move(str(lbl), str(val_lbls / lbl.name))

    print(f"[FIX] Split complete: {len(images) - split_count} train, {split_count} val")


def list_datasets() -> None:
    print("\nAvailable datasets:\n")
    for name, info in DATASETS.items():
        loc = Path(info["location"])
        yaml_path = loc / "data.yaml"
        status = "downloaded" if yaml_path.exists() else "not downloaded"

        classes_info = ""
        if yaml_path.exists():
            with open(yaml_path) as f:
                cfg = yaml.safe_load(f)
            names = cfg.get("names", [])
            if isinstance(names, dict):
                names = list(names.values())
            classes_info = ", ".join(str(n) for n in names)
            classes_info = f"\n  {'':25s} classes: {classes_info}"

        print(f"  {name:<25s} {info['description']}")
        print(f"  {'':25s} location: {loc}  ({status}){classes_info}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download military detection datasets")
    parser.add_argument("--dataset", help="Dataset name or 'all'")
    parser.add_argument("--list", action="store_true", dest="show_list", help="List datasets")
    args = parser.parse_args()

    if args.show_list or (not args.dataset):
        list_datasets()
        if not args.dataset:
            print("Use --dataset <name> to download, or --dataset all for everything.")
        return

    if args.dataset == "all":
        for name in DATASETS:
            download_one(name)
        print("\nAll datasets downloaded.")
    elif args.dataset in DATASETS:
        download_one(args.dataset)
    else:
        print(f"[ERROR] Unknown dataset '{args.dataset}'")
        print(f"Available: {', '.join(DATASETS.keys())}")
        sys.exit(1)


if __name__ == "__main__":
    main()
