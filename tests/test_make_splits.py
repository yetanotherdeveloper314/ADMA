"""Small integration tests for the ADMA split builder using a temporary dataset."""

from __future__ import annotations

import csv
import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_splits.py"


class MakeSplitsIntegrationTests(unittest.TestCase):
    def _make_fixture(self, root: Path) -> None:
        (root / "src" / "adma").mkdir(parents=True)
        (root / "src" / "adma" / "datasets.py").write_text(
            "LABEL_MAP = {'Tank': 'tank', 'Armored': 'armored_vehicle'}\n"
            "TARGET_CLASSES = ('armored_vehicle', 'tank')\n"
            "EXCLUDED_DATASETS = frozenset()\n"
            "UNRESOLVED_LABELS = frozenset()\n"
            "NEGATIVE_REVIEW_CSV = 'data/negative_review.csv'\n",
            encoding="utf-8",
        )
        datasets = ("tank_images", "military_vehicles", "military_vehicles_obj", "military_objects")
        duplicate_rows: list[dict[str, str]] = []
        group_num = 0
        for dataset in datasets:
            dataset_root = root / "data" / dataset
            yaml_text = (
                "path: .\n"
                "train: train/images\n"
                "val: valid/images\n"
                "test: test/images\n"
                "nc: 2\n"
                "names:\n"
                "  - Tank\n"
                "  - Armored\n"
            )
            (dataset_root / "data.yaml").parent.mkdir(parents=True, exist_ok=True)
            (dataset_root / "data.yaml").write_text(yaml_text, encoding="utf-8")
            for split in ("train", "valid", "test"):
                image_dir = dataset_root / split / "images"
                label_dir = dataset_root / split / "labels"
                image_dir.mkdir(parents=True, exist_ok=True)
                label_dir.mkdir(parents=True, exist_ok=True)
                for index in range(6):
                    filename = f"{split}_{index}.jpg"
                    # Minimal byte content is sufficient for split logic; no image decoding occurs here.
                    (image_dir / filename).write_bytes(b"synthetic-image")
                    class_id = index % 2
                    (label_dir / f"{Path(filename).stem}.txt").write_text(
                        f"{class_id} 0.5 0.5 0.2 0.2\n", encoding="utf-8"
                    )
                    group_num += 1
                    group_id = f"group_{group_num:06d}"
                    # Four source files are exact duplicates for split purposes: they must stay together.
                    if split == "train" and index == 0:
                        group_id = "shared_cross_dataset_group"
                    duplicate_rows.append(
                        {
                            "group_id": group_id,
                            "dataset": dataset,
                            "split": "valid" if split == "valid" else split,
                            "image_path": (
                                Path("data") / dataset / split / "images" / filename
                            ).as_posix(),
                        }
                    )
        dup_path = root / "data" / "splits" / "duplicate_groups.csv"
        dup_path.parent.mkdir(parents=True, exist_ok=True)
        with dup_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["group_id", "dataset", "split", "image_path"]
            )
            writer.writeheader()
            writer.writerows(duplicate_rows)

    def test_missing_target_class_does_not_create_test_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._make_fixture(root)
            policy_path = root / "src" / "adma" / "datasets.py"
            policy_path.write_text(
                "LABEL_MAP = {'Tank': 'tank'}\n"
                "TARGET_CLASSES = ('armored_vehicle', 'tank')\n"
                "EXCLUDED_DATASETS = frozenset()\n"
                "UNRESOLVED_LABELS = frozenset()\n"
                "NEGATIVE_REVIEW_CSV = 'data/negative_review.csv'\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--repo-root", str(root)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("Target classes with no eligible positive annotations", result.stderr)
            output = root / "data" / "combined_v1"
            self.assertFalse((output / "test_images.txt").exists())
            self.assertTrue(
                (root / "data" / "splits" / "make_splits_preflight_summary.csv").is_file()
            )
            self.assertTrue((root / "data" / "splits" / "make_splits_label_review.csv").is_file())

    def test_builds_deterministic_split_and_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._make_fixture(root)
            command = [sys.executable, str(SCRIPT), "--repo-root", str(root)]
            first = subprocess.run(command, check=True, capture_output=True, text=True)
            output = root / "data" / "combined_v1"
            test_list = output / "test_images.txt"
            fingerprint = output / "test_images.sha256"
            self.assertTrue(test_list.is_file())
            self.assertTrue(fingerprint.is_file())
            expected = hashlib.sha256(test_list.read_bytes()).hexdigest()
            self.assertEqual(fingerprint.read_text(encoding="utf-8").split()[0], expected)
            self.assertIn("Duplicate groups spanning multiple assigned splits: 0", first.stdout)
            self.assertTrue((output / "images" / "train").is_dir())
            self.assertTrue((output / "images" / "val").is_dir())
            self.assertTrue((output / "images" / "test").is_dir())
            self.assertTrue((output / "labels" / "test").is_dir())
            with (output / "split_manifest.csv").open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            shared_group_rows = [
                row for row in rows if row["group_id"] == "shared_cross_dataset_group"
            ]
            self.assertEqual(len(shared_group_rows), 4)
            self.assertEqual(len({row["assigned_split"] for row in shared_group_rows}), 1)
            for group_id in {row["group_id"] for row in rows}:
                split_names = {row["assigned_split"] for row in rows if row["group_id"] == group_id}
                self.assertEqual(len(split_names), 1)

            # A second run must not silently overwrite the locked output.
            second = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("Refusing to alter a locked dataset", second.stderr)

            # Force rebuilding is allowed when the proposed test list remains unchanged.
            rebuilt = subprocess.run(
                command + ["--force-rebuild"], check=True, capture_output=True, text=True
            )
            rebuilt_hash = hashlib.sha256(test_list.read_bytes()).hexdigest()
            self.assertEqual(rebuilt_hash, expected)
            self.assertIn(expected, rebuilt.stdout)


if __name__ == "__main__":
    unittest.main()
