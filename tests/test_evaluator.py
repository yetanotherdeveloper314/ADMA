"""Unit tests for the pure functions in evaluator.py."""

import unittest
from pathlib import Path

import numpy as np

from scripts.evaluator import (
    compute_average_precision,
    compute_image_id,
    compute_iou,
    compute_ious,
    compute_precision_recall_f1,
    images_dir_to_labels_dir,
    match_predictions_to_ground_truth,
    predictions_from_detector,
)


class TestComputeIoU(unittest.TestCase):
    def test_identical_boxes(self):
        box = [0, 0, 10, 10]
        self.assertAlmostEqual(compute_iou(box, box), 1.0)

    def test_no_overlap(self):
        self.assertEqual(
            compute_iou(
                [0, 0, 10, 10],
                [20, 20, 30, 30],
            ),
            0.0,
        )

    def test_partial_overlap(self):
        iou = compute_iou(
            [0, 0, 10, 10],
            [5, 5, 15, 15],
        )

        self.assertAlmostEqual(iou, 25 / 175)


class TestComputeIoUs(unittest.TestCase):
    def test_matches_scalar_iou(self):
        box = [0, 0, 10, 10]
        boxes = [
            [0, 0, 10, 10],
            [5, 5, 15, 15],
            [20, 20, 30, 30],
        ]

        result = compute_ious(box, boxes)

        expected = np.array([compute_iou(box, other) for other in boxes])

        np.testing.assert_allclose(result, expected)

    def test_empty_boxes(self):
        result = compute_ious(
            [0, 0, 10, 10],
            [],
        )

        self.assertEqual(result.size, 0)


class TestComputeImageId(unittest.TestCase):
    def test_relative_image_id(self):
        root = Path("data/combined").resolve()
        image = root / "images/val/tank_images/tank001.jpg"

        self.assertEqual(
            compute_image_id(image, root),
            "images/val/tank_images/tank001",
        )

    def test_same_filename_different_folders(self):
        root = Path("data/combined").resolve()

        image_a = root / "images/val/tank_images/img001.jpg"
        image_b = root / "images/val/military_vehicles/img001.jpg"

        self.assertNotEqual(
            compute_image_id(image_a, root),
            compute_image_id(image_b, root),
        )


class TestImagesDirToLabelsDir(unittest.TestCase):
    def test_replaces_images_component(self):
        result = images_dir_to_labels_dir(Path("data/images/val"))

        self.assertEqual(
            result,
            Path("data/labels/val"),
        )

    def test_does_not_replace_images_substring(self):
        result = images_dir_to_labels_dir(Path("data/images/val/tank_images"))

        self.assertEqual(
            result,
            Path("data/labels/val/tank_images"),
        )


class TestPredictionsFromDetector(unittest.TestCase):
    def test_uses_given_class_mapping(self):
        mapping = {
            "drone": 1,
            "tank": 13,
        }

        results = {
            "img001": [
                {
                    "class": "tank",
                    "confidence": 0.9,
                    "bbox": [0, 0, 10, 10],
                }
            ]
        }

        predictions = predictions_from_detector(
            results,
            mapping,
        )

        self.assertEqual(
            predictions[0]["class_id"],
            13,
        )
        self.assertEqual(
            predictions[0]["image_id"],
            "img001",
        )
        self.assertAlmostEqual(
            predictions[0]["score"],
            0.9,
        )

    def test_unknown_class_raises_error(self):
        results = {
            "img001": [
                {
                    "class": "unknown",
                    "confidence": 0.9,
                    "bbox": [0, 0, 10, 10],
                }
            ]
        }

        with self.assertRaises(ValueError):
            predictions_from_detector(
                results,
                {"tank": 13},
            )

    def test_empty_mapping_raises_error(self):
        with self.assertRaises(ValueError):
            predictions_from_detector(
                [],
                {},
            )


class TestComputeAveragePrecision(unittest.TestCase):
    def test_no_ground_truth(self):
        ap, precision, recall = compute_average_precision(
            np.array([1.0]),
            np.array([0.0]),
            n_gt=0,
        )

        self.assertIsNone(ap)
        self.assertEqual(precision, 0.0)
        self.assertEqual(recall, 0.0)

    def test_perfect_detector(self):
        tp = np.array([1.0, 1.0, 1.0])
        fp = np.array([0.0, 0.0, 0.0])

        ap, precision, recall = compute_average_precision(
            tp,
            fp,
            n_gt=3,
        )

        self.assertAlmostEqual(ap, 1.0)
        self.assertAlmostEqual(precision, 1.0)
        self.assertAlmostEqual(recall, 1.0)

    def test_all_false_positives(self):
        tp = np.array([0.0, 0.0])
        fp = np.array([1.0, 1.0])

        ap, precision, recall = compute_average_precision(
            tp,
            fp,
            n_gt=2,
        )

        self.assertAlmostEqual(ap, 0.0)
        self.assertAlmostEqual(precision, 0.0)
        self.assertAlmostEqual(recall, 0.0)

    def test_coco_101_point_partial_recall(self):
        # One TP out of two GT objects gives maximum recall = 0.5.
        # COCO samples 101 recall points:
        # 51 points have precision 1, so AP = 51 / 101.
        tp = np.array([1.0])
        fp = np.array([0.0])

        ap, precision, recall = compute_average_precision(
            tp,
            fp,
            n_gt=2,
        )

        self.assertAlmostEqual(
            ap,
            51 / 101,
            places=5,
        )
        self.assertAlmostEqual(precision, 1.0)
        self.assertAlmostEqual(recall, 0.5)


class TestPrecisionRecallF1(unittest.TestCase):
    def test_perfect_score(self):
        precision, recall, f1 = compute_precision_recall_f1(
            tp=10,
            fp=0,
            fn=0,
        )

        self.assertEqual(
            (precision, recall, f1),
            (1.0, 1.0, 1.0),
        )

    def test_known_values(self):
        precision, recall, f1 = compute_precision_recall_f1(
            tp=3,
            fp=1,
            fn=2,
        )

        self.assertAlmostEqual(precision, 0.75)
        self.assertAlmostEqual(recall, 0.6)

        expected_f1 = 2 * precision * recall / (precision + recall)

        self.assertAlmostEqual(f1, expected_f1)

    def test_all_zero(self):
        self.assertEqual(
            compute_precision_recall_f1(
                tp=0,
                fp=0,
                fn=0,
            ),
            (0.0, 0.0, 0.0),
        )


class TestMatchPredictionsToGroundTruth(unittest.TestCase):
    def test_correct_match(self):
        ground_truth = [(0, [0, 0, 10, 10])]

        predictions = [(0, [0, 0, 10, 10], 0.9)]

        result = match_predictions_to_ground_truth(
            ground_truth,
            predictions,
            iou_thresh=0.5,
        )

        self.assertEqual(
            result,
            [(0, 0)],
        )

    def test_misclassification(self):
        ground_truth = [(13, [0, 0, 10, 10])]

        predictions = [(1, [0, 0, 10, 10], 0.9)]

        result = match_predictions_to_ground_truth(
            ground_truth,
            predictions,
            iou_thresh=0.5,
        )

        self.assertEqual(
            result,
            [(13, 1)],
        )

    def test_false_negative(self):
        ground_truth = [(13, [0, 0, 10, 10])]

        result = match_predictions_to_ground_truth(
            ground_truth,
            [],
            iou_thresh=0.5,
        )

        self.assertEqual(
            result,
            [(13, None)],
        )

    def test_false_positive(self):
        predictions = [(13, [0, 0, 10, 10], 0.9)]

        result = match_predictions_to_ground_truth(
            [],
            predictions,
            iou_thresh=0.5,
        )

        self.assertEqual(
            result,
            [(None, 13)],
        )


if __name__ == "__main__":
    unittest.main()
