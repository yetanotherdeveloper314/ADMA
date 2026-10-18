"""Focused unit tests for the duplicate grouping core."""

import unittest
from pathlib import Path

from scripts.find_duplicates import BKTree, ImageRecord, build_components


class BKTreeTests(unittest.TestCase):
    def test_radius_query_uses_hamming_distance(self) -> None:
        tree = BKTree()
        tree.add(0b0)
        tree.add(0b11111)  # distance 5 from 0
        tree.add(0b111111)  # distance 6 from 0

        found = {value: distance for value, distance in tree.query(0, radius=5)}
        self.assertEqual(found[0], 0)
        self.assertEqual(found[0b11111], 5)
        self.assertNotIn(0b111111, found)


class GroupingTests(unittest.TestCase):
    def test_filename_and_phash_matches_are_merged(self) -> None:
        records = [
            ImageRecord("tank_images", "train", Path("a.jpg"), None),
            ImageRecord("tank_images", "valid", Path("b.jpg"), None),
            ImageRecord("military_objects", "test", Path("photo_jpg.rf.aaa.jpg"), "photo_jpg"),
            ImageRecord("military_vehicles", "train", Path("photo_jpg.rf.bbb.jpg"), "photo_jpg"),
        ]
        # Hashes 0 and 31 differ by exactly five bits; the two Roboflow files
        # share a source-name key but have hashes far apart from the first pair.
        hashes = [0, 31, 0xFFFFFFFFFFFFFFFF, 0xAAAAAAAAAAAAAAAA]
        components = build_components(records, hashes, threshold=5, seed=42)
        groups = {frozenset(group) for group in components}
        self.assertEqual(
            groups,
            {frozenset({0, 1}), frozenset({2, 3})},
        )


if __name__ == "__main__":
    unittest.main()
