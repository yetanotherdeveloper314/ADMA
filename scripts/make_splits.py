#!/usr/bin/env python3
"""Create a group-safe 70/15/15 ADMA dataset and lock its test-image list.

Prerequisites:
    poetry run python scripts/find_duplicates.py --threshold 5

The duplicate CSV must cover every source image. Images in a duplicate group are
assigned to the same split, including when that group spans source datasets.
Source labels are normalized through ``src/adma/datasets.py``. Images with
missing/invalid labels, unknown or unresolved labels, or unapproved empty labels
are reported to ``label_review.csv`` and not copied. The existing policy's
``EXCLUDED_DATASETS`` is honored; groups from excluded sources are still used
when enforcing duplicate-group boundaries.

By default, an existing output directory is never overwritten. To rebuild it,
use ``--force-rebuild``. If the proposed test path list differs from the saved
list, rebuilding is blocked unless ``--allow-test-list-change`` is also passed
(for an explicitly approved test-set reset).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import random
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - environment-dependent
    raise SystemExit("Missing PyYAML. Install project dependencies with `poetry install`.") from exc

DATASET_NAMES = (
    "tank_images",
    "military_vehicles",
    "military_vehicles_obj",
    "military_objects",
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CANONICAL_SPLIT = {"val": "valid", "valid": "valid", "validation": "valid"}
SPLITS = ("train", "val", "test")
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


@dataclass(frozen=True)
class Policy:
    label_map: dict[str, str | None]
    target_classes: tuple[str, ...]
    excluded_datasets: frozenset[str]
    unresolved_labels: frozenset[str]
    negative_review_csv: str


@dataclass
class SourceImage:
    dataset: str
    source_split: str
    image_path: Path
    image_root: Path
    labels_root: Path
    class_names: list[str]
    group_id: str = ""
    remapped_label_text: str | None = None
    classes: set[str] = field(default_factory=set)
    instance_counts: Counter[str] = field(default_factory=Counter)
    review_reason: str | None = None
    source_label_names: set[str] = field(default_factory=set)
    approved_negative: bool = False
    output_relative_image: str | None = None

    @property
    def label_path(self) -> Path:
        return self.labels_root / self.image_path.relative_to(self.image_root).with_suffix(".txt")

    @property
    def included(self) -> bool:
        return self.remapped_label_text is not None and self.review_reason is None


@dataclass
class Group:
    group_id: str
    records: list[SourceImage]
    per_dataset: Counter[str] = field(default_factory=Counter)
    eligible_per_dataset: Counter[str] = field(default_factory=Counter)
    per_dataset_class_instances: Counter[tuple[str, str]] = field(default_factory=Counter)
    classes: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        for record in self.records:
            # Split ratios are based on the complete source pool, including images later held
            # for review by the approved label policy. This also assigns excluded-source groups.
            self.per_dataset[record.dataset] += 1
            if not record.included:
                continue
            self.eligible_per_dataset[record.dataset] += 1
            self.classes.update(record.classes)
            for name, count in record.instance_counts.items():
                self.per_dataset_class_instances[(record.dataset, name)] += count


def parse_args() -> argparse.Namespace:
    script_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Create a duplicate-group-safe 70/15/15 ADMA dataset and test-list fingerprint."
    )
    parser.add_argument("--repo-root", type=Path, default=script_root)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument(
        "--duplicates-csv",
        type=Path,
        default=None,
        help="Defaults to data/splits/duplicate_groups.csv (from find_duplicates.py).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Defaults to data/combined_v1."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild an existing output directory; does not by itself permit test-list changes.",
    )
    parser.add_argument(
        "--allow-test-list-change",
        action="store_true",
        help="Permit changing an existing test image list. Use only with mentor approval.",
    )
    return parser.parse_args()


def load_policy(repo_root: Path) -> Policy:
    """Load the project's approved label policy without hard-coding a second copy."""
    policy_path = repo_root / "src" / "adma" / "datasets.py"
    if not policy_path.is_file():
        raise FileNotFoundError(
            f"Could not find {policy_path}. Run this script from the ADMA repository root."
        )
    spec = importlib.util.spec_from_file_location("adma_split_policy", policy_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load label policy from {policy_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = ("LABEL_MAP", "TARGET_CLASSES")
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise ValueError(f"{policy_path} is missing required constants: {', '.join(missing)}")
    return Policy(
        label_map=dict(module.LABEL_MAP),
        target_classes=tuple(module.TARGET_CLASSES),
        excluded_datasets=frozenset(getattr(module, "EXCLUDED_DATASETS", ())),
        unresolved_labels=frozenset(getattr(module, "UNRESOLVED_LABELS", ())),
        negative_review_csv=str(getattr(module, "NEGATIVE_REVIEW_CSV", "data/negative_review.csv")),
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return data


def _names_from_yaml(raw_names: Any, yaml_path: Path) -> list[str]:
    if isinstance(raw_names, list) and all(isinstance(item, str) for item in raw_names):
        return raw_names
    if isinstance(raw_names, dict):
        try:
            return [str(raw_names[key]) for key in sorted(raw_names, key=lambda item: int(item))]
        except (ValueError, TypeError):
            raise ValueError(f"Class names in {yaml_path} must use numeric keys starting at 0")
    raise ValueError(f"No usable `names` list/dict in {yaml_path}")


def _path_from_yaml(
    value: str, dataset_root: Path, config_path: Path, repo_root: Path
) -> Path | None:
    """Resolve a split path, preferring the local dataset over stale export paths."""
    raw = str(value).strip()
    if not raw:
        return None
    normalized = raw.replace("\\", "/")
    candidate_path = Path(normalized)
    if candidate_path.is_absolute() and candidate_path.is_dir():
        return candidate_path.resolve()

    # A YAML may contain an absolute Windows path copied from another machine.
    win_path = PureWindowsPath(raw)
    if win_path.is_absolute():
        parts = list(win_path.parts)
        lower_parts = [part.lower() for part in parts]
        if dataset_root.name.lower() in lower_parts:
            pos = lower_parts.index(dataset_root.name.lower())
            tail = parts[pos + 1 :]
            candidate = dataset_root.joinpath(*tail)
            if candidate.is_dir():
                return candidate.resolve()
        # If the YAML refers to a standard split (train/images etc), try that suffix.
        for split in ("train", "valid", "val", "test"):
            for index, part in enumerate(lower_parts):
                if part == split:
                    candidate = dataset_root.joinpath(*parts[index:])
                    if candidate.is_dir():
                        return candidate.resolve()

    relative = Path(normalized)
    candidates = [dataset_root / relative, config_path.parent / relative, repo_root / relative]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    return None


def _choose_image_dir(split_dir: Path) -> Path:
    child_images = split_dir / "images"
    if child_images.is_dir():
        return child_images.resolve()
    return split_dir.resolve()


def _find_source_split_dirs(
    dataset_root: Path, repo_root: Path
) -> tuple[list[tuple[str, Path, list[str]]], list[str]]:
    warnings: list[str] = []
    yaml_path = dataset_root / "data.yaml"
    if not yaml_path.is_file():
        yaml_path = dataset_root / "data.yml"
    config: dict[str, Any] = {}
    if yaml_path.is_file():
        config = _load_yaml(yaml_path)
        class_names = _names_from_yaml(config.get("names"), yaml_path)
    else:
        raise FileNotFoundError(f"Missing data.yaml/data.yml for {dataset_root}")

    split_dirs: list[tuple[str, Path, list[str]]] = []
    keys_seen: set[tuple[str, str]] = set()
    for raw_key in ("train", "val", "valid", "test"):
        value = config.get(raw_key)
        if value is None:
            continue
        values = [value] if isinstance(value, str) else value if isinstance(value, list) else []
        canonical = CANONICAL_SPLIT.get(raw_key, raw_key)
        for raw_value in values:
            if not isinstance(raw_value, str):
                continue
            resolved = _path_from_yaml(raw_value, dataset_root, yaml_path, repo_root)
            if resolved is None:
                continue
            image_dir = _choose_image_dir(resolved)
            key = (canonical, str(image_dir.resolve()))
            if key in keys_seen:
                continue
            keys_seen.add(key)
            split_dirs.append((canonical, image_dir, class_names))

    # Standard Roboflow layout is a fallback for incomplete/stale YAML entries.
    conventional = {
        "train": [dataset_root / "train" / "images", dataset_root / "train"],
        "valid": [
            dataset_root / "valid" / "images",
            dataset_root / "val" / "images",
            dataset_root / "valid",
            dataset_root / "val",
        ],
        "test": [dataset_root / "test" / "images", dataset_root / "test"],
    }
    existing = {split for split, _folder, _names in split_dirs}
    for split, options in conventional.items():
        if split in existing:
            continue
        candidate = next((path for path in options if path.is_dir()), None)
        if candidate is not None:
            split_dirs.append((split, _choose_image_dir(candidate), class_names))

    if not split_dirs:
        warnings.append(f"No image split directories resolved for {dataset_root}")
    return split_dirs, warnings


def discover_images(repo_root: Path, data_root: Path) -> tuple[list[SourceImage], list[str]]:
    records: list[SourceImage] = []
    warnings: list[str] = []
    seen: dict[tuple[str, str], str] = {}
    for dataset in DATASET_NAMES:
        dataset_root = data_root / dataset
        if not dataset_root.is_dir():
            raise FileNotFoundError(
                f"Required source dataset directory does not exist: {dataset_root}"
            )
        split_dirs, split_warnings = _find_source_split_dirs(dataset_root, repo_root)
        warnings.extend(split_warnings)
        if not split_dirs:
            raise FileNotFoundError(f"No train/val/test folders found under {dataset_root}")
        found = 0
        for split, image_root, class_names in split_dirs:
            labels_root = (
                image_root.parent / "labels"
                if image_root.name.lower() == "images"
                else image_root / "labels"
            )
            for path in sorted(image_root.rglob("*"), key=lambda item: item.as_posix().casefold()):
                if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                resolved = path.resolve()
                key = (dataset, str(resolved))
                if key in seen:
                    previous_split = seen[key]
                    if previous_split != split:
                        raise ValueError(
                            f"The same source file is configured in multiple source splits: {path} "
                            f"({previous_split} and {split}). Fix the source data.yaml first."
                        )
                    continue
                seen[key] = split
                records.append(
                    SourceImage(
                        dataset=dataset,
                        source_split=split,
                        image_path=resolved,
                        image_root=image_root.resolve(),
                        labels_root=labels_root.resolve(),
                        class_names=class_names,
                    )
                )
                found += 1
        if found == 0:
            warnings.append(f"No supported image files found under {dataset_root}")
    records.sort(
        key=lambda item: (item.dataset, item.source_split, item.image_path.as_posix().casefold())
    )
    if not records:
        raise FileNotFoundError("No images were discovered in the four expected datasets.")
    return records, warnings


def _normalise_manifest_path(raw_path: str, dataset: str, repo_root: Path, data_root: Path) -> Path:
    raw = raw_path.strip()
    normalized = raw.replace("\\", "/")
    direct = Path(normalized)
    if direct.is_absolute() and direct.is_file():
        return direct.resolve()
    if not PureWindowsPath(raw).is_absolute():
        for candidate in (repo_root / direct, data_root / direct, repo_root / "data" / direct):
            if candidate.is_file():
                return candidate.resolve()

    # Remap foreign absolute paths (usually Windows paths) to this clone's data root.
    parts = (
        list(PureWindowsPath(raw).parts)
        if PureWindowsPath(raw).is_absolute()
        else normalized.split("/")
    )
    lower_parts = [part.lower() for part in parts]
    dataset_lower = dataset.lower()
    if dataset_lower in lower_parts:
        position = lower_parts.index(dataset_lower)
        candidate = data_root / dataset / Path(*parts[position + 1 :])
        if candidate.is_file():
            return candidate.resolve()

    # Relative repo paths usually begin with data/<dataset>/...
    for index, part in enumerate(lower_parts):
        if part == dataset_lower:
            candidate = data_root / dataset / Path(*parts[index + 1 :])
            if candidate.is_file():
                return candidate.resolve()
    raise FileNotFoundError(
        f"Could not resolve image_path {raw_path!r} from duplicate manifest for dataset {dataset!r}. "
        "Regenerate data/splits/duplicate_groups.csv against the current data/ tree."
    )


def apply_duplicate_groups(
    records: list[SourceImage], csv_path: Path, repo_root: Path, data_root: Path
) -> None:
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Missing {csv_path}. Run `poetry run python scripts/find_duplicates.py --threshold 5` first."
        )
    record_by_key = {
        (record.dataset, str(record.image_path.resolve())): record for record in records
    }
    seen_manifest_keys: set[tuple[str, str]] = set()
    group_by_key: dict[tuple[str, str], str] = {}
    manifest_keys: set[tuple[str, str]] = set()
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"group_id", "dataset", "split", "image_path"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{csv_path} must have columns: {', '.join(sorted(required))}")
        for row in reader:
            dataset = str(row.get("dataset", "")).strip()
            if dataset not in DATASET_NAMES:
                raise ValueError(f"Unexpected dataset name {dataset!r} in {csv_path}")
            path = _normalise_manifest_path(
                str(row.get("image_path", "")), dataset, repo_root, data_root
            )
            key = (dataset, str(path.resolve()))
            if key in seen_manifest_keys:
                # find_duplicates may list the same physical file more than once if split YAML aliases overlap.
                existing = group_by_key[key]
                new = str(row.get("group_id", "")).strip()
                if existing != new:
                    raise ValueError(
                        f"Image {path} appears in multiple duplicate groups: {existing}, {new}"
                    )
                continue
            seen_manifest_keys.add(key)
            manifest_keys.add(key)
            group_id = str(row.get("group_id", "")).strip()
            if not group_id:
                raise ValueError(f"Empty group_id for {path} in {csv_path}")
            group_by_key[key] = group_id
            record = record_by_key.get(key)
            if record is None:
                raise ValueError(
                    f"Duplicate manifest image was not discovered from its source YAML: {dataset}: {path}. "
                    "Regenerate the duplicate manifest after fixing data.yaml."
                )
            record.group_id = group_id

    ungrouped = [record for record in records if not record.group_id]
    manifest_only_missing = set(record_by_key) - manifest_keys
    if ungrouped or manifest_only_missing:
        preview = [f"{record.dataset}: {record.image_path}" for record in ungrouped[:5]]
        raise ValueError(
            f"Duplicate manifest does not cover every discovered image. Missing {len(ungrouped)} image(s). "
            + "; ".join(preview)
            + ". Rerun scripts/find_duplicates.py; do not split without a complete manifest."
        )


def _read_negative_approvals(repo_root: Path, policy: Policy) -> set[tuple[str, str, str]]:
    manifest = Path(policy.negative_review_csv)
    if not manifest.is_absolute():
        manifest = repo_root / manifest
    if not manifest.is_file():
        return set()
    required = {"dataset", "split", "image_filename", "verified_no_target"}
    approved: set[tuple[str, str, str]] = set()
    with manifest.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{manifest} must contain columns: {', '.join(sorted(required))}")
        for row in reader:
            value = str(row.get("verified_no_target", "")).strip().lower()
            if value in {"yes", "true", "1", "y"}:
                split = str(row.get("split", "")).strip().lower()
                split = CANONICAL_SPLIT.get(split, split)
                approved.add(
                    (
                        str(row.get("dataset", "")).strip(),
                        split,
                        str(row.get("image_filename", "")).strip(),
                    )
                )
    return approved


def prepare_labels(
    records: list[SourceImage], policy: Policy, repo_root: Path
) -> list[dict[str, str]]:
    """Validate source label files and remap annotation IDs; return review rows."""
    review_rows: list[dict[str, str]] = []
    class_to_id = {name: index for index, name in enumerate(policy.target_classes)}
    approvals = _read_negative_approvals(repo_root, policy)
    for record in records:
        if record.dataset in policy.excluded_datasets:
            record.review_reason = "dataset_excluded_by_policy"
            review_rows.append(_review_row(record))
            continue
        record.approved_negative = (
            record.dataset,
            record.source_split,
            record.image_path.name,
        ) in approvals
        label_path = record.label_path
        if not label_path.is_file():
            record.review_reason = "missing_label_file"
            review_rows.append(_review_row(record))
            continue
        try:
            lines = label_path.read_text(encoding="utf-8-sig").splitlines()
        except OSError as exc:
            record.review_reason = f"cannot_read_label_file:{exc}"
            review_rows.append(_review_row(record))
            continue

        output_lines: list[str] = []
        source_label_names: set[str] = set()
        target_present = False
        invalid_reason: str | None = None
        for line_no, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                invalid_reason = f"invalid_yolo_row_at_line_{line_no}"
                break
            try:
                class_id = int(parts[0])
            except ValueError:
                invalid_reason = f"invalid_class_id_at_line_{line_no}"
                break
            if class_id < 0 or class_id >= len(record.class_names):
                invalid_reason = f"class_id_out_of_range_at_line_{line_no}"
                break
            raw_name = record.class_names[class_id]
            source_label_names.add(raw_name)
            if raw_name not in policy.label_map:
                invalid_reason = f"unmapped_source_label:{raw_name}"
                break
            mapped_name = policy.label_map[raw_name]
            if mapped_name is None:
                continue
            if mapped_name not in class_to_id:
                invalid_reason = f"mapped_class_not_in_target_taxonomy:{mapped_name}"
                break
            target_present = True
            out_parts = list(parts)
            out_parts[0] = str(class_to_id[mapped_name])
            output_lines.append(" ".join(out_parts))
            record.classes.add(mapped_name)
            record.instance_counts[mapped_name] += 1

        record.source_label_names = source_label_names
        if invalid_reason:
            record.review_reason = invalid_reason
        elif source_label_names.intersection(policy.unresolved_labels):
            if record.approved_negative and not target_present:
                output_lines = []
                record.classes.clear()
                record.instance_counts.clear()
            else:
                unresolved = ";".join(
                    sorted(source_label_names.intersection(policy.unresolved_labels))
                )
                record.review_reason = f"unresolved_label:{unresolved}"
        elif not output_lines and not record.approved_negative:
            reason = (
                "source_empty_label_file"
                if not source_label_names
                else "only_out_of_scope_labels_remaining"
            )
            record.review_reason = f"negative_review_required:{reason}"

        if record.review_reason is None:
            record.remapped_label_text = "\n".join(output_lines) + ("\n" if output_lines else "")
        else:
            review_rows.append(_review_row(record))
    return review_rows


def _review_row(record: SourceImage) -> dict[str, str]:
    return {
        "dataset": record.dataset,
        "source_split": record.source_split,
        "image_filename": record.image_path.name,
        "image_path": record.image_path.as_posix(),
        "reason": record.review_reason or "unspecified",
        "source_labels": ";".join(sorted(record.source_label_names)),
    }


def group_records(records: list[SourceImage]) -> list[Group]:
    by_group: dict[str, list[SourceImage]] = defaultdict(list)
    for record in records:
        if not record.group_id:
            raise ValueError(f"No duplicate group assigned to {record.image_path}")
        by_group[record.group_id].append(record)
    groups = [Group(group_id, members) for group_id, members in by_group.items()]
    groups.sort(key=lambda group: group.group_id)
    return groups


def _assignment_loss(
    assignment: dict[str, str],
    groups: list[Group],
    included_by_group: dict[str, Group],
    dataset_totals: Counter[str],
    class_totals: Counter[tuple[str, str]],
    ratios: dict[str, float],
) -> float:
    image_counts: dict[tuple[str, str], int] = defaultdict(int)
    class_counts: dict[tuple[str, str, str], int] = defaultdict(int)
    for group_id, split in assignment.items():
        group = included_by_group[group_id]
        for dataset, count in group.per_dataset.items():
            image_counts[(dataset, split)] += count
        for (dataset, class_name), count in group.per_dataset_class_instances.items():
            class_counts[(dataset, class_name, split)] += count

    loss = 0.0
    for dataset, total in dataset_totals.items():
        for split, ratio in ratios.items():
            target = total * ratio
            actual = image_counts[(dataset, split)]
            loss += 2.0 * ((actual - target) / max(total, 1)) ** 2
    # A softer per-source class-instance term helps preserve rare classes while prioritizing image ratios.
    for (dataset, class_name), total in class_totals.items():
        denom = max(total, 1)
        for split, ratio in ratios.items():
            target = total * ratio
            actual = class_counts[(dataset, class_name, split)]
            loss += 0.20 * ((actual - target) / denom) ** 2
    return loss


def assign_groups(groups: list[Group], seed: int, ratios: dict[str, float]) -> dict[str, str]:
    """Greedy deterministic group-level assignment with source and class balancing."""
    included_groups = [group for group in groups if group.eligible_per_dataset]
    if not included_groups:
        raise ValueError("No images have eligible labels under the approved taxonomy policy.")
    included_by_group = {group.group_id: group for group in groups}

    # Balance every source's entire image pool, not merely the subset that survives label review.
    dataset_totals: Counter[str] = Counter()
    class_totals: Counter[tuple[str, str]] = Counter()
    class_group_ids: dict[str, set[str]] = defaultdict(set)
    for group in groups:
        dataset_totals.update(group.per_dataset)
        class_totals.update(group.per_dataset_class_instances)
        for class_name in group.classes:
            class_group_ids[class_name].add(group.group_id)

    # Larger groups and rare-class groups are placed first; the seeded shuffle only resolves ties.
    rng = random.Random(seed)
    tie_break = {group.group_id: rng.random() for group in groups}
    class_freq = Counter(class_name for group in groups for class_name in group.classes)

    def priority(group: Group) -> tuple[float, int, float, str]:
        rarity = sum(1.0 / max(class_freq[class_name], 1) for class_name in group.classes)
        image_count = sum(group.per_dataset.values())
        return (-rarity, -image_count, tie_break[group.group_id], group.group_id)

    ordered = sorted(groups, key=priority)
    assignment: dict[str, str] = {}
    for group in ordered:
        best_split: str | None = None
        best_loss: float | None = None
        for split in SPLITS:
            candidate = dict(assignment)
            candidate[group.group_id] = split
            loss = _assignment_loss(
                candidate, groups, included_by_group, dataset_totals, class_totals, ratios
            )
            # Split order is stable for deterministic tie behavior.
            if best_loss is None or loss < best_loss - 1e-12:
                best_loss, best_split = loss, split
        assert best_split is not None
        assignment[group.group_id] = best_split

    # Try to ensure each observed class occurs in both val and test if at least two independent
    # duplicate groups contain it. Never break already-achieved coverage for another such class.
    feasible_classes = {
        class_name for class_name, group_ids in class_group_ids.items() if len(group_ids) >= 2
    }

    def class_group_count(class_name: str, split: str, hypothetical: dict[str, str]) -> int:
        return sum(
            1 for group_id in class_group_ids[class_name] if hypothetical.get(group_id) == split
        )

    def move_preserves_coverage(group_id: str, target_split: str) -> bool:
        assignment[group_id]
        hypothetical = dict(assignment)
        hypothetical[group_id] = target_split
        for class_name in feasible_classes:
            for split in ("val", "test"):
                before = class_group_count(class_name, split, assignment)
                after = class_group_count(class_name, split, hypothetical)
                if before > 0 and after == 0:
                    return False
        return True

    for _ in range(max(1, len(feasible_classes) * 4)):
        missing = [
            (class_name, split)
            for class_name in sorted(feasible_classes)
            for split in ("val", "test")
            if class_group_count(class_name, split, assignment) == 0
        ]
        if not missing:
            break
        changed = False
        for class_name, target_split in missing:
            candidates: list[tuple[float, str]] = []
            for group_id in sorted(class_group_ids[class_name]):
                if assignment[group_id] == target_split:
                    continue
                if not move_preserves_coverage(group_id, target_split):
                    continue
                hypothetical = dict(assignment)
                hypothetical[group_id] = target_split
                loss = _assignment_loss(
                    hypothetical, groups, included_by_group, dataset_totals, class_totals, ratios
                )
                candidates.append((loss, group_id))
            if candidates:
                _, selected = min(candidates, key=lambda pair: (pair[0], pair[1]))
                assignment[selected] = target_split
                changed = True
        if not changed:
            break

    return assignment


def _output_relative(record: SourceImage) -> str:
    rel = record.image_path.relative_to(record.image_root)
    # Keeping source split as a subfolder prevents same-named files from original splits colliding.
    return (Path(record.dataset) / f"source_{record.source_split}" / rel).as_posix()


def _counts_for_output(
    records: list[SourceImage], assignment: dict[str, str]
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    copied = [record for record in records if record.included]
    assigned_split_images: Counter[str] = Counter()
    assigned_source_split_images: Counter[tuple[str, str]] = Counter()
    split_images: Counter[str] = Counter()
    source_split_images: Counter[tuple[str, str]] = Counter()
    class_images: Counter[tuple[str, str]] = Counter()
    class_instances: Counter[tuple[str, str]] = Counter()
    for record in records:
        split = assignment[record.group_id]
        assigned_split_images[split] += 1
        assigned_source_split_images[(split, record.dataset)] += 1
    for record in copied:
        split = assignment[record.group_id]
        split_images[split] += 1
        source_split_images[(split, record.dataset)] += 1
        for class_name in record.classes:
            class_images[(split, class_name)] += 1
        for class_name, count in record.instance_counts.items():
            class_instances[(split, class_name)] += count

    rows: list[dict[str, str]] = []
    for split in SPLITS:
        rows.append(
            {
                "section": "assigned_split_total",
                "split": split,
                "dataset": "ALL",
                "class": "ALL",
                "images": str(assigned_split_images[split]),
                "instances": "",
            }
        )
        for dataset in DATASET_NAMES:
            rows.append(
                {
                    "section": "source_assignment",
                    "split": split,
                    "dataset": dataset,
                    "class": "ALL",
                    "images": str(assigned_source_split_images[(split, dataset)]),
                    "instances": "",
                }
            )
        rows.append(
            {
                "section": "copied_split_total",
                "split": split,
                "dataset": "ALL",
                "class": "ALL",
                "images": str(split_images[split]),
                "instances": str(
                    sum(
                        count for (sp, _class_name), count in class_instances.items() if sp == split
                    )
                ),
            }
        )
        for dataset in DATASET_NAMES:
            rows.append(
                {
                    "section": "source_split",
                    "split": split,
                    "dataset": dataset,
                    "class": "ALL",
                    "images": str(source_split_images[(split, dataset)]),
                    "instances": "",
                }
            )
        for class_name in sorted(
            {name for _split, name in class_images} | {name for _split, name in class_instances}
        ):
            rows.append(
                {
                    "section": "class",
                    "split": split,
                    "dataset": "ALL",
                    "class": class_name,
                    "images": str(class_images[(split, class_name)]),
                    "instances": str(class_instances[(split, class_name)]),
                }
            )
    summary = {
        "assigned_split_images": dict(assigned_split_images),
        "assigned_source_split_images": {
            f"{split}/{dataset}": count
            for (split, dataset), count in assigned_source_split_images.items()
        },
        "split_images": dict(split_images),
        "source_split_images": {
            f"{split}/{dataset}": count for (split, dataset), count in source_split_images.items()
        },
        "class_images": {
            f"{split}/{class_name}": count for (split, class_name), count in class_images.items()
        },
        "class_instances": {
            f"{split}/{class_name}": count for (split, class_name), count in class_instances.items()
        },
    }
    return rows, summary


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _candidate_test_list(records: list[SourceImage], assignment: dict[str, str]) -> list[str]:
    paths = []
    for record in records:
        if record.included and assignment[record.group_id] == "test":
            paths.append((Path("images") / "test" / _output_relative(record)).as_posix())
    return sorted(paths)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_existing_test_lock(
    output_dir: Path, proposed_test_list: list[str], force_rebuild: bool, allow_change: bool
) -> None:
    if not output_dir.exists():
        return
    if not force_rebuild:
        raise FileExistsError(
            f"Output already exists: {output_dir}. Refusing to alter a locked dataset. "
            "Use --force-rebuild only when you intend to rebuild it."
        )
    old_list_path = output_dir / "test_images.txt"
    if old_list_path.is_file():
        old_list = old_list_path.read_text(encoding="utf-8").splitlines()
        if old_list != proposed_test_list and not allow_change:
            raise PermissionError(
                "The proposed test image list differs from the existing lock. Do not change test images "
                "without mentor approval. After approval, pass --allow-test-list-change explicitly."
            )
    elif not allow_change:
        raise PermissionError(
            "Existing output has no test_images.txt lock file. Use --allow-test-list-change only "
            "after confirming the test-set reset is approved."
        )


def _write_output(
    output_dir: Path,
    records: list[SourceImage],
    groups: list[Group],
    assignment: dict[str, str],
    policy: Policy,
    review_rows: list[dict[str, str]],
    seed: int,
    ratios: dict[str, float],
    repo_root: Path,
) -> tuple[str, dict[str, Any]]:
    staging_dir = output_dir.with_name(output_dir.name + ".tmp-build")
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    all_group_splits: dict[str, set[str]] = defaultdict(set)
    for group in groups:
        if group.group_id in assignment:
            all_group_splits[group.group_id].add(assignment[group.group_id])
    leaking_groups = {
        group_id: splits for group_id, splits in all_group_splits.items() if len(splits) > 1
    }
    if leaking_groups:
        raise AssertionError(f"Duplicate group leakage detected: {leaking_groups}")

    included = [record for record in records if record.included]
    used_destinations: set[str] = set()
    for record in included:
        split = assignment[record.group_id]
        relative = _output_relative(record)
        image_rel = (Path("images") / split / relative).as_posix()
        label_rel = (Path("labels") / split / Path(relative).with_suffix(".txt")).as_posix()
        if image_rel in used_destinations:
            raise ValueError(
                f"Output filename collision for {image_rel}. Check source split folders."
            )
        used_destinations.add(image_rel)
        record.output_relative_image = image_rel
        image_destination = staging_dir / image_rel
        label_destination = staging_dir / label_rel
        image_destination.parent.mkdir(parents=True, exist_ok=True)
        label_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(record.image_path, image_destination)
        label_destination.write_text(record.remapped_label_text or "", encoding="utf-8")

    data_yaml = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(policy.target_classes),
        "names": list(policy.target_classes),
    }
    with (staging_dir / "data.yaml").open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(
            data_yaml, handle, default_flow_style=False, sort_keys=False, allow_unicode=True
        )

    test_list = _candidate_test_list(records, assignment)
    (staging_dir / "test_images.txt").write_text(
        "".join(path + "\n" for path in test_list), encoding="utf-8", newline="\n"
    )
    digest = _sha256_file(staging_dir / "test_images.txt")
    (staging_dir / "test_images.sha256").write_text(
        f"{digest}  test_images.txt\n", encoding="utf-8", newline="\n"
    )

    _write_csv(
        staging_dir / "label_review.csv",
        ["dataset", "source_split", "image_filename", "image_path", "reason", "source_labels"],
        review_rows,
    )
    count_rows, summary = _counts_for_output(records, assignment)
    _write_csv(
        staging_dir / "split_summary.csv",
        ["section", "split", "dataset", "class", "images", "instances"],
        count_rows,
    )
    manifest_rows = []
    for record in sorted(
        records, key=lambda item: (item.dataset, item.source_split, item.image_path.as_posix())
    ):
        manifest_rows.append(
            {
                "group_id": record.group_id,
                "dataset": record.dataset,
                "source_split": record.source_split,
                "source_image_path": record.image_path.as_posix(),
                "assigned_split": assignment.get(record.group_id, "not_copied"),
                "included": "yes" if record.included else "no",
                "reason": record.review_reason or "",
                "output_image_path": record.output_relative_image or "",
            }
        )
    _write_csv(
        staging_dir / "split_manifest.csv",
        [
            "group_id",
            "dataset",
            "source_split",
            "source_image_path",
            "assigned_split",
            "included",
            "reason",
            "output_image_path",
        ],
        manifest_rows,
    )

    lock_meta = {
        "test_list_sha256": digest,
        "test_images": len(test_list),
        "seed": seed,
        "ratios": ratios,
        "target_classes": list(policy.target_classes),
        "dataset_names": list(DATASET_NAMES),
        "test_list_file": "test_images.txt",
        "note": "The test image path list is locked; correcting a test label must not alter this list.",
    }
    with (staging_dir / "split_lock.yaml").open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(
            lock_meta, handle, default_flow_style=False, sort_keys=False, allow_unicode=True
        )

    # Swap only after every output and fingerprint has been written successfully.
    backup_dir = output_dir.with_name(output_dir.name + ".previous")
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    if output_dir.exists():
        output_dir.rename(backup_dir)
    try:
        staging_dir.rename(output_dir)
    except OSError:
        if backup_dir.exists() and not output_dir.exists():
            backup_dir.rename(output_dir)
        raise
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    return digest, summary


def main() -> int:
    args = parse_args()
    ratios = {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio}
    if (
        any(value <= 0 or value >= 1 for value in ratios.values())
        or abs(sum(ratios.values()) - 1.0) > 1e-9
    ):
        print(
            "ERROR: train/val/test ratios must each be between 0 and 1 and sum to 1.",
            file=sys.stderr,
        )
        return 2

    repo_root = args.repo_root.expanduser().resolve()
    data_root = args.data_root or (repo_root / "data")
    data_root = data_root.expanduser()
    if not data_root.is_absolute():
        data_root = repo_root / data_root
    data_root = data_root.resolve()
    duplicate_csv = args.duplicates_csv or (data_root / "splits" / "duplicate_groups.csv")
    if not duplicate_csv.is_absolute():
        duplicate_csv = repo_root / duplicate_csv
    duplicate_csv = duplicate_csv.resolve()
    output_dir = args.output_dir or (data_root / "combined_v1")
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir = output_dir.expanduser().resolve()

    try:
        policy = load_policy(repo_root)
        records, warnings = discover_images(repo_root, data_root)
        for warning in warnings:
            print(f"WARNING: {warning}")
        print(f"Discovered source images: {len(records):,}")
        apply_duplicate_groups(records, duplicate_csv, repo_root, data_root)
        review_rows = prepare_labels(records, policy, repo_root)
        groups = group_records(records)
        included_before = sum(1 for record in records if record.included)
        if included_before == 0:
            raise ValueError("No images are eligible after applying the approved label policy.")
        assignment = assign_groups(groups, args.seed, ratios)
        eligible_classes = {
            name for record in records if record.included for name in record.classes
        }
        classes_with_two_groups = {
            name
            for name in eligible_classes
            if len(
                {
                    record.group_id
                    for record in records
                    if record.included and name in record.classes
                }
            )
            >= 2
        }
        missing_coverage = {
            (name, split)
            for name in classes_with_two_groups
            for split in ("val", "test")
            if not any(
                record.included and name in record.classes and assignment[record.group_id] == split
                for record in records
            )
        }
        absent_classes = set(policy.target_classes) - eligible_classes
        single_group_classes = eligible_classes - classes_with_two_groups
        if missing_coverage or single_group_classes or absent_classes:
            # Preserve diagnostics outside combined_v1, so a failed preflight never creates or
            # changes the locked test list but still gives the user actionable review files.
            preflight_dir = data_root / "splits"
            preflight_dir.mkdir(parents=True, exist_ok=True)
            review_path = preflight_dir / "make_splits_label_review.csv"
            _write_csv(
                review_path,
                [
                    "dataset",
                    "source_split",
                    "image_filename",
                    "image_path",
                    "reason",
                    "source_labels",
                ],
                review_rows,
            )
            count_rows, _preflight_summary = _counts_for_output(records, assignment)
            summary_path = preflight_dir / "make_splits_preflight_summary.csv"
            _write_csv(
                summary_path,
                ["section", "split", "dataset", "class", "images", "instances"],
                count_rows,
            )
            print(
                "ERROR: refusing to create/lock a test set because class coverage is incomplete.",
                file=sys.stderr,
            )
            if missing_coverage:
                print(
                    "Observed classes missing from validation/test despite multiple independent groups:",
                    file=sys.stderr,
                )
                for class_name, split in sorted(missing_coverage):
                    print(f"  - {class_name}: missing from {split}", file=sys.stderr)
            if single_group_classes:
                print(
                    "Classes that occur in only one duplicate group (cannot be in both val and test):",
                    file=sys.stderr,
                )
                print("  " + ", ".join(sorted(single_group_classes)), file=sys.stderr)
            if absent_classes:
                print("Target classes with no eligible positive annotations:", file=sys.stderr)
                print("  " + ", ".join(sorted(absent_classes)), file=sys.stderr)
            print(f"Label review: {review_path}", file=sys.stderr)
            print(f"Preflight counts: {summary_path}", file=sys.stderr)
            print(
                "Review the source annotations and src/adma/datasets.py before trying again. "
                "No new output or test fingerprint was written; any existing locked output was left unchanged.",
                file=sys.stderr,
            )
            return 1

        proposed_test_list = _candidate_test_list(records, assignment)
        _check_existing_test_lock(
            output_dir, proposed_test_list, args.force_rebuild, args.allow_test_list_change
        )

        digest, summary = _write_output(
            output_dir,
            records,
            groups,
            assignment,
            policy,
            review_rows,
            args.seed,
            ratios,
            repo_root,
        )

        print("\nASSIGNMENT COUNTS (all source images, before label-policy exclusions)")
        for dataset in DATASET_NAMES:
            counts = [
                summary["assigned_source_split_images"].get(f"{split}/{dataset}", 0)
                for split in SPLITS
            ]
            total = sum(counts)
            percentages = [100 * value / total if total else 0.0 for value in counts]
            targets = [total * ratios[split] for split in SPLITS]
            print(
                f"{dataset:24} train={counts[0]:5,} ({percentages[0]:5.1f}%, target {targets[0]:.1f})  "
                f"val={counts[1]:5,} ({percentages[1]:5.1f}%, target {targets[1]:.1f})  "
                f"test={counts[2]:5,} ({percentages[2]:5.1f}%, target {targets[2]:.1f})"
            )
        print("\nSPLIT SUMMARY (eligible copied images)")
        print("split  total_images")
        for split in SPLITS:
            print(f"{split:5}  {summary['split_images'].get(split, 0):12,}")
        print("\nPER-SOURCE IMAGE COUNTS")
        for dataset in DATASET_NAMES:
            counts = [
                summary["source_split_images"].get(f"{split}/{dataset}", 0) for split in SPLITS
            ]
            print(f"{dataset:24} train={counts[0]:5,}  val={counts[1]:5,}  test={counts[2]:5,}")
        print("\nPER-CLASS COUNTS (images containing class / object instances)")
        all_classes = list(policy.target_classes)
        for class_name in all_classes:
            display = []
            for split in SPLITS:
                image_count = summary["class_images"].get(f"{split}/{class_name}", 0)
                instance_count = summary["class_instances"].get(f"{split}/{class_name}", 0)
                display.append(f"{split}={image_count} img/{instance_count} obj")
            print(f"{class_name:22} " + "  ".join(display))

        duplicate_splits: dict[str, set[str]] = defaultdict(set)
        for group in groups:
            if group.group_id in assignment:
                duplicate_splits[group.group_id].add(assignment[group.group_id])
        leaking = [group_id for group_id, values in duplicate_splits.items() if len(values) > 1]
        print("\nINTEGRITY CHECKS")
        print(f"Duplicate groups spanning multiple assigned splits: {len(leaking)}")
        print(
            f"Source images discovered: {len(records):,}; eligible images copied: {included_before:,}"
        )
        print(f"Images held for label review or excluded by policy: {len(review_rows):,}")
        print(
            "Class coverage check: every target class appears in both validation and test across independent groups."
        )
        print(f"\nTest image list: {output_dir / 'test_images.txt'}")
        print(f"SHA-256: {digest}")
        print(f"Fingerprint file: {output_dir / 'test_images.sha256'}")
        print(f"Split manifest: {output_dir / 'split_manifest.csv'}")
        print(f"Counts table: {output_dir / 'split_summary.csv'}")
        print(f"Label review: {output_dir / 'label_review.csv'}")
        print(
            "\nLock check: changes to the test path list are blocked on rebuild unless explicitly approved."
        )
        if leaking:
            return 2
    except (OSError, ValueError, PermissionError, ImportError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
