#!/usr/bin/env python3
"""Find duplicate and near-duplicate images across the ADMA source datasets.

The script combines two independent signals:
1. Roboflow filenames sharing the prefix before ``.rf.``.
2. 64-bit perceptual hashes whose Hamming distance is <= ``--threshold``.

Both types of match are combined with connected components. The CSV includes
all discovered images, including singleton groups, so downstream split code can
assign every image a group without inventing a separate ID for unmatched files.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable

try:
    import imagehash
    import yaml
    from PIL import Image, ImageDraw, ImageOps
except ImportError as exc:  # pragma: no cover - depends on project environment
    missing = getattr(exc, "name", None) or str(exc)
    print(
        f"Missing dependency ({missing}). From the repository root run:\n"
        "    poetry add --group dev imagehash\n"
        "If PyYAML or Pillow is missing too, install the project's dependencies. ",
        file=sys.stderr,
    )
    raise SystemExit(2) from exc


DATASET_NAMES = (
    "tank_images",
    "military_vehicles",
    "military_vehicles_obj",
    "military_objects",
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLIT_KEYS = ("train", "val", "valid", "test")
SPLIT_ORDER = {"train": 0, "valid": 1, "val": 1, "test": 2}


@dataclass(frozen=True)
class ImageRecord:
    dataset: str
    split: str
    path: Path
    roboflow_key: str | None


@dataclass
class BKNode:
    value: int
    children: dict[int, "BKNode"]


class BKTree:
    """BK-tree for exact-radius queries under Hamming distance."""

    def __init__(self) -> None:
        self.root: BKNode | None = None

    @staticmethod
    def distance(left: int, right: int) -> int:
        return (left ^ right).bit_count()

    def add(self, value: int) -> None:
        if self.root is None:
            self.root = BKNode(value=value, children={})
            return

        node = self.root
        while True:
            distance = self.distance(value, node.value)
            if distance == 0:
                return
            child = node.children.get(distance)
            if child is None:
                node.children[distance] = BKNode(value=value, children={})
                return
            node = child

    def query(self, value: int, radius: int) -> list[tuple[int, int]]:
        """Return (stored_hash, Hamming distance) pairs within ``radius``."""
        if self.root is None:
            return []

        matches: list[tuple[int, int]] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            distance = self.distance(value, node.value)
            if distance <= radius:
                matches.append((node.value, distance))

            lower = distance - radius
            upper = distance + radius
            for edge_distance, child in node.children.items():
                if lower <= edge_distance <= upper:
                    stack.append(child)
        return matches


class UnionFind:
    def __init__(self, count: int) -> None:
        self.parent = list(range(count))
        self.rank = [0] * count

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def parse_args() -> argparse.Namespace:
    script_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Group exact and perceptual-hash duplicates across ADMA datasets."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=script_root,
        help="Repository root (defaults to the parent of scripts/).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Dataset root (defaults to <repo-root>/data).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV output (defaults to data/splits/duplicate_groups.csv).",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=5,
        help="Maximum pHash Hamming distance for a near-duplicate match (default: 5).",
    )
    parser.add_argument(
        "--review-samples",
        type=int,
        default=30,
        help="Create contact sheets for this many randomly sampled duplicate groups (default: 30).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for repeatable review samples."
    )
    parser.add_argument(
        "--review-dir",
        type=Path,
        default=None,
        help="Contact-sheet output directory (defaults to data/splits/duplicate_review).",
    )
    return parser.parse_args()


def _is_windows_absolute(value: str) -> bool:
    return PureWindowsPath(value).is_absolute()


def _as_local_path(value: str | Path, relative_to: Path | None = None) -> Path | None:
    """Resolve a path if it is usable on this OS; ignore foreign Windows paths on POSIX."""
    text_value = str(value)
    candidate = Path(text_value)
    if candidate.is_absolute():
        return candidate
    if _is_windows_absolute(text_value):
        return candidate if os.name == "nt" else None
    if relative_to is not None:
        return (relative_to / candidate).resolve()
    # Keep a relative split entry relative so it can be tried against each YAML root.
    return candidate


def _load_yaml(yaml_path: Path) -> dict[str, Any]:
    with yaml_path.open("r", encoding="utf-8-sig") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a YAML mapping in {yaml_path}")
    return loaded


def _find_yaml(dataset_root: Path) -> Path | None:
    preferred = (dataset_root / "data.yaml", dataset_root / "data.yml")
    for candidate in preferred:
        if candidate.is_file():
            return candidate
    remaining = sorted(
        [*dataset_root.glob("*.yaml"), *dataset_root.glob("*.yml")],
        key=lambda p: p.name.lower(),
    ) if dataset_root.is_dir() else []
    return remaining[0] if remaining else None


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        try:
            identity = str(path.resolve())
        except OSError:
            identity = str(path)
        if identity not in seen:
            seen.add(identity)
            result.append(path)
    return result


def _root_candidates(
    config: dict[str, Any], yaml_path: Path | None, dataset_root: Path, repo_root: Path
) -> list[Path]:
    roots: list[Path] = []
    # Prefer the dataset's actual folder under data/. Roboflow YAML files can
    # retain an absolute path from the machine where they were originally exported.
    if yaml_path is not None:
        roots.append(yaml_path.parent)
    roots.append(dataset_root)
    configured = config.get("path")
    if configured:
        resolved = _as_local_path(str(configured), yaml_path.parent if yaml_path else dataset_root)
        if resolved is not None:
            roots.append(resolved)
    roots.append(repo_root / "data")
    return _unique_paths(roots)


def _split_value_to_paths(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def _resolve_split_dirs(
    dataset_name: str,
    dataset_root: Path,
    yaml_path: Path | None,
    config: dict[str, Any],
    repo_root: Path,
) -> list[tuple[str, Path]]:
    roots = _root_candidates(config, yaml_path, dataset_root, repo_root)
    resolved: list[tuple[str, Path]] = []

    for key in SPLIT_KEYS:
        raw_value = config.get(key)
        if raw_value is None:
            continue
        canonical_split = "valid" if key == "val" else key
        values = _split_value_to_paths(raw_value)
        for value in values:
            value_path = _as_local_path(value)
            candidates: list[Path] = []
            if value_path is not None and value_path.is_absolute():
                candidates.append(value_path)
            else:
                candidates.extend((root / value for root in roots))
            selected = next((candidate for candidate in candidates if candidate.is_dir()), None)
            if selected is not None:
                resolved.append((canonical_split, selected))

    # Some exports may have incomplete YAML or no YAML. Fall back to conventional folders.
    conventional = {
        "train": (dataset_root / "train" / "images", dataset_root / "train"),
        "valid": (
            dataset_root / "valid" / "images",
            dataset_root / "val" / "images",
            dataset_root / "valid",
            dataset_root / "val",
        ),
        "test": (dataset_root / "test" / "images", dataset_root / "test"),
    }
    existing_splits = {split for split, _ in resolved}
    for split, options in conventional.items():
        if split not in existing_splits:
            candidate = next((path for path in options if path.is_dir()), None)
            if candidate is not None:
                resolved.append((split, candidate))

    # Keep each dataset/split/folder once if YAML aliases point at the same directory.
    deduped: list[tuple[str, Path]] = []
    seen: set[tuple[str, str]] = set()
    for split, folder in resolved:
        key = (split, str(folder.resolve()))
        if key not in seen:
            seen.add(key)
            deduped.append((split, folder))
    return deduped


def _roboflow_key(path: Path) -> str | None:
    filename = path.name
    marker = ".rf."
    if marker not in filename.lower():
        return None
    prefix = filename.lower().split(marker, maxsplit=1)[0].strip()
    return prefix or None


def discover_images(repo_root: Path, data_root: Path) -> tuple[list[ImageRecord], list[str]]:
    records: list[ImageRecord] = []
    warnings: list[str] = []
    seen_records: set[tuple[str, str, str]] = set()

    for dataset_name in DATASET_NAMES:
        dataset_root = data_root / dataset_name
        if not dataset_root.is_dir():
            warnings.append(f"Dataset folder not found: {dataset_root}")
            continue

        yaml_path = _find_yaml(dataset_root)
        config: dict[str, Any] = {}
        if yaml_path is not None:
            try:
                config = _load_yaml(yaml_path)
            except (OSError, ValueError, yaml.YAMLError) as exc:
                warnings.append(f"Could not read {yaml_path}: {exc}; using conventional folders.")
        else:
            warnings.append(f"No data.yaml/data.yml found under {dataset_root}; using conventional folders.")

        split_dirs = _resolve_split_dirs(dataset_name, dataset_root, yaml_path, config, repo_root)
        if not split_dirs:
            warnings.append(f"No train/valid/test image folders found for {dataset_name}.")
            continue

        found_for_dataset = 0
        for split, image_dir in split_dirs:
            for image_path in sorted(image_dir.rglob("*")):
                if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                try:
                    resolved_path = image_path.resolve()
                except OSError:
                    resolved_path = image_path
                record_key = (dataset_name, split, str(resolved_path))
                if record_key in seen_records:
                    continue
                seen_records.add(record_key)
                records.append(
                    ImageRecord(
                        dataset=dataset_name,
                        split=split,
                        path=resolved_path,
                        roboflow_key=_roboflow_key(image_path),
                    )
                )
                found_for_dataset += 1
        if found_for_dataset == 0:
            warnings.append(f"No supported image files found for {dataset_name}.")

    records.sort(
        key=lambda record: (
            record.dataset,
            SPLIT_ORDER.get(record.split, 99),
            str(record.path).casefold(),
        )
    )
    return records, warnings


def compute_phashes(records: list[ImageRecord]) -> tuple[list[int | None], list[str]]:
    hashes: list[int | None] = []
    errors: list[str] = []
    for record in records:
        try:
            with Image.open(record.path) as image:
                phash = imagehash.phash(image.convert("RGB"))
            hashes.append(int(str(phash), 16))
        except Exception as exc:  # malformed/corrupt images should not abort the full scan
            hashes.append(None)
            errors.append(f"{record.path}: {type(exc).__name__}: {exc}")
    return hashes, errors


def build_components(
    records: list[ImageRecord], hashes: list[int | None], threshold: int, seed: int
) -> list[list[int]]:
    union_find = UnionFind(len(records))

    # First merge Roboflow images sharing the source filename prefix before '.rf.'.
    filename_groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        if record.roboflow_key:
            filename_groups[record.roboflow_key].append(index)
    for member_indices in filename_groups.values():
        first = member_indices[0]
        for other in member_indices[1:]:
            union_find.union(first, other)

    # Exact hashes are buckets, so each unique 64-bit hash enters the BK-tree only once.
    hash_groups: dict[int, list[int]] = defaultdict(list)
    for index, hash_value in enumerate(hashes):
        if hash_value is not None:
            hash_groups[hash_value].append(index)
    hash_representatives = {value: indices[0] for value, indices in hash_groups.items()}
    for indices in hash_groups.values():
        for other in indices[1:]:
            union_find.union(indices[0], other)

    unique_hashes = list(hash_groups)
    random.Random(seed).shuffle(unique_hashes)
    tree = BKTree()
    for hash_value in unique_hashes:
        tree.add(hash_value)

    # Search the BK-tree for each unique hash. Candidate pairs are processed once.
    for hash_value in unique_hashes:
        for candidate, _distance in tree.query(hash_value, threshold):
            if candidate > hash_value:
                union_find.union(
                    hash_representatives[hash_value], hash_representatives[candidate]
                )

    components_by_root: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        components_by_root[union_find.find(index)].append(index)

    components = list(components_by_root.values())
    components.sort(
        key=lambda members: min(
            (records[index].dataset, records[index].split, str(records[index].path).casefold())
            for index in members
        )
    )
    return components


def display_path(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except (OSError, ValueError):
        return str(path)


def write_groups_csv(
    output_path: Path,
    records: list[ImageRecord],
    components: list[list[int]],
    repo_root: Path,
) -> list[tuple[str, list[int]]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    identified: list[tuple[str, list[int]]] = []
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("group_id", "dataset", "split", "image_path"))
        for group_number, members in enumerate(components, start=1):
            group_id = f"group_{group_number:06d}"
            identified.append((group_id, members))
            for index in sorted(
                members,
                key=lambda idx: (
                    records[idx].dataset,
                    SPLIT_ORDER.get(records[idx].split, 99),
                    str(records[idx].path).casefold(),
                ),
            ):
                record = records[index]
                writer.writerow(
                    (group_id, record.dataset, record.split, display_path(record.path, repo_root))
                )
    return identified


def _short_label(text: str, width: int = 34, max_lines: int = 2) -> str:
    lines = textwrap.wrap(text, width=width, break_long_words=True, break_on_hyphens=True)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][: max(0, width - 3)] + "..."
    return "\n".join(lines)


def create_review_sheets(
    identified_groups: list[tuple[str, list[int]]],
    records: list[ImageRecord],
    repo_root: Path,
    review_dir: Path,
    sample_count: int,
    seed: int,
) -> list[tuple[str, int, str]]:
    duplicate_groups = [(group_id, members) for group_id, members in identified_groups if len(members) > 1]
    sample_size = min(max(sample_count, 0), len(duplicate_groups))
    sampled = random.Random(seed).sample(duplicate_groups, sample_size) if sample_size else []

    review_dir.mkdir(parents=True, exist_ok=True)
    # Remove stale sheets generated by this script, but preserve unrelated files.
    for old_sheet in review_dir.glob("group_*.jpg"):
        try:
            old_sheet.unlink()
        except OSError:
            pass

    manifest: list[tuple[str, int, str]] = []
    columns = 3
    tile_width = 280
    tile_height = 235
    thumbnail_size = (tile_width - 20, 175)
    max_images_per_sheet = 24

    for group_id, all_members in sampled:
        members = all_members[:max_images_per_sheet]
        rows = (len(members) + columns - 1) // columns
        canvas = Image.new("RGB", (columns * tile_width, max(rows, 1) * tile_height + 38), "white")
        draw = ImageDraw.Draw(canvas)
        header = f"{group_id}: showing {len(members)} of {len(all_members)} images"
        draw.text((10, 8), header, fill="black")

        for position, record_index in enumerate(members):
            record = records[record_index]
            column = position % columns
            row = position // columns
            x0 = column * tile_width
            y0 = row * tile_height + 38
            try:
                with Image.open(record.path) as source:
                    thumbnail = ImageOps.contain(source.convert("RGB"), thumbnail_size)
                paste_x = x0 + (tile_width - thumbnail.width) // 2
                paste_y = y0 + max(0, (180 - thumbnail.height) // 2)
                canvas.paste(thumbnail, (paste_x, paste_y))
            except Exception as exc:
                draw.text((x0 + 8, y0 + 8), f"Could not open image: {exc}", fill="red")

            label = _short_label(
                f"{record.dataset} / {record.split}\n{record.path.name}", width=38, max_lines=3
            )
            draw.multiline_text((x0 + 7, y0 + 181), label, fill="black", spacing=2)

        sheet_path = review_dir / f"{group_id}.jpg"
        canvas.save(sheet_path, quality=90)
        manifest.append((group_id, len(all_members), display_path(sheet_path, repo_root)))

    manifest_path = review_dir / "review_samples.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("group_id", "group_size", "contact_sheet_path"))
        writer.writerows(manifest)
    return manifest


def main() -> int:
    args = parse_args()
    if not 0 <= args.threshold <= 64:
        print("ERROR: --threshold must be between 0 and 64 for a 64-bit pHash.", file=sys.stderr)
        return 2
    if args.review_samples < 0:
        print("ERROR: --review-samples cannot be negative.", file=sys.stderr)
        return 2

    repo_root = args.repo_root.expanduser().resolve()
    data_root = (args.data_root or (repo_root / "data")).expanduser()
    if not data_root.is_absolute():
        data_root = repo_root / data_root
    data_root = data_root.resolve()
    output_path = args.output or (data_root / "splits" / "duplicate_groups.csv")
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    output_path = output_path.resolve()
    review_dir = args.review_dir or (output_path.parent / "duplicate_review")
    if not review_dir.is_absolute():
        review_dir = repo_root / review_dir
    review_dir = review_dir.resolve()

    print("ADMA duplicate-image scan")
    print(f"Repository root: {repo_root}")
    print(f"Data root:       {data_root}")
    print(f"pHash threshold: {args.threshold} (Hamming distance <= threshold)")
    print("Scanning datasets: " + ", ".join(DATASET_NAMES))

    records, discovery_warnings = discover_images(repo_root, data_root)
    for warning in discovery_warnings:
        print(f"WARNING: {warning}")
    if not records:
        print("ERROR: no image files were found. Check data-root and each dataset's data.yaml.", file=sys.stderr)
        return 2

    print(f"Computing perceptual hashes for {len(records):,} images...")
    hashes, hash_errors = compute_phashes(records)
    components = build_components(records, hashes, args.threshold, args.seed)
    identified_groups = write_groups_csv(output_path, records, components, repo_root)
    review_manifest = create_review_sheets(
        identified_groups, records, repo_root, review_dir, args.review_samples, args.seed
    )

    multi_split_groups = 0
    multi_dataset_groups = 0
    duplicate_groups = 0
    duplicate_images = 0
    for _group_id, members in identified_groups:
        if len(members) > 1:
            duplicate_groups += 1
            duplicate_images += len(members)
            if len({records[index].split for index in members}) > 1:
                multi_split_groups += 1
            if len({records[index].dataset for index in members}) > 1:
                multi_dataset_groups += 1

    print("\nSUMMARY")
    print(f"Images scanned:                    {len(records):,}")
    print(f"Images successfully hashed:        {len(records) - len(hash_errors):,}")
    print(f"Connected groups (including singletons): {len(identified_groups):,}")
    print(f"Duplicate groups (size > 1):        {duplicate_groups:,}")
    print(f"Images in duplicate groups:         {duplicate_images:,}")
    print(f"Groups spanning multiple splits (leaks): {multi_split_groups:,}")
    print(f"Groups spanning multiple datasets:  {multi_dataset_groups:,}")
    print(f"Output CSV: {display_path(output_path, repo_root)}")
    print(f"Review sheets: {display_path(review_dir, repo_root)}")
    print(f"Review sheets created:              {len(review_manifest):,}")
    if hash_errors:
        print(f"Images whose pHash could not be read: {len(hash_errors):,}")
        for error in hash_errors[:10]:
            print(f"  WARNING: {error}")
        if len(hash_errors) > 10:
            print(f"  ... and {len(hash_errors) - 10:,} more; these images remain in the CSV.")
    if discovery_warnings:
        print(f"Dataset discovery warnings:         {len(discovery_warnings):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
