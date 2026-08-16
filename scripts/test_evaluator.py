"""
Unit tests for the pure functions in evaluator.py.

Run with:
    python -m unittest test_evaluator.py -v
or, if pytest is available:
    pytest test_evaluator.py -v

Only the pure, side-effect-free functions are covered here (no file I/O,
no OpenCV, no argparse): compute_iou, compute_image_id,
images_dir_to_labels_dir, compute_average_precision,
compute_precision_recall_f1, and match_predictions_to_ground_truth.
"""

import unittest
from pathlib import Path

import numpy as np

from evaluator import (
    compute_average_precision,
    compute_image_id,
    compute_iou,
    compute_precision_recall_f1,
    images_dir_to_labels_dir,
    match_predictions_to_ground_truth,
)


class TestComputeIoU(unittest.TestCase):
    def test_identical_boxes(self):
        box = [10, 10, 50, 50]
        self.assertAlmostEqual(compute_iou(box, box), 1.0)

    def test_no_overlap(self):
        self.assertEqual(compute_iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)

    def test_touching_edges_no_overlap(self):
        # boxes that share an edge but have zero area of intersection
        self.assertEqual(compute_iou([0, 0, 10, 10], [10, 0, 20, 10]), 0.0)

    def test_partial_overlap_known_value(self):
        # box_a: 0,0 - 10,10 (area 100)
        # box_b: 5,5 - 15,15 (area 100)
        # intersection: 5,5 - 10,10 (area 25)
        # union: 100 + 100 - 25 = 175
        # iou = 25/175
        iou = compute_iou([0, 0, 10, 10], [5, 5, 15, 15])
        self.assertAlmostEqual(iou, 25 / 175)

    def test_one_box_fully_inside_other(self):
        outer = [0, 0, 20, 20]  # area 400
        inner = [5, 5, 10, 10]  # area 25
        # intersection == inner area == 25, union == 400
        self.assertAlmostEqual(compute_iou(outer, inner), 25 / 400)

    def test_zero_area_box(self):
        self.assertEqual(compute_iou([5, 5, 5, 5], [0, 0, 10, 10]), 0.0)


class TestComputeImageId(unittest.TestCase):
    def test_simple_relative_path(self):
        root = Path("/data/combined")
        img = Path("/data/combined/images/val/tank_images/tank001.jpg")
        self.assertEqual(compute_image_id(img, root), "images/val/tank_images/tank001")

    def test_strips_extension_regardless_of_type(self):
        root = Path("/data/combined")
        img_png = Path("/data/combined/images/val/img1.png")
        img_jpg = Path("/data/combined/images/val/img1.jpg")
        self.assertEqual(compute_image_id(img_png, root), "images/val/img1")
        self.assertEqual(compute_image_id(img_jpg, root), "images/val/img1")

    def test_disambiguates_same_filename_in_sibling_folders(self):
        # the core motivating case: two different images, same filename,
        # different subfolders -- must NOT produce the same id
        root = Path("/data/combined")
        img_a = Path("/data/combined/images/val/tank_images/img001.jpg")
        img_b = Path("/data/combined/images/val/military_vehicles/img001.jpg")
        id_a = compute_image_id(img_a, root)
        id_b = compute_image_id(img_b, root)
        self.assertNotEqual(id_a, id_b)

    def test_fallback_when_not_under_root(self):
        # image_path isn't under dataset_root -- falls back to filename only
        root = Path("/some/other/root")
        img = Path("/data/combined/images/val/tank001.jpg")
        self.assertEqual(compute_image_id(img, root), "tank001")

    def test_root_equal_to_images_dir_reduces_to_stem(self):
        # backward-compatible default: dataset_root == images_dir means
        # the id collapses to the old stem-only behavior
        images_dir = Path("/data/combined/images/val/tank_images")
        img = images_dir / "tank001.jpg"
        self.assertEqual(compute_image_id(img, images_dir), "tank001")


class TestImagesDirToLabelsDir(unittest.TestCase):
    def test_simple_swap(self):
        self.assertEqual(
            images_dir_to_labels_dir(Path("data/images/val")),
            Path("data/labels/val"),
        )

    def test_does_not_mangle_substring_match(self):
        # regression test for the real bug: 'tank_images' contains the
        # substring 'images' but must NOT be rewritten
        result = images_dir_to_labels_dir(Path("data/images/val/tank_images"))
        self.assertEqual(result, Path("data/labels/val/tank_images"))

    def test_multiple_source_folder_names_untouched(self):
        for folder in ["tank_images", "military_vehicles", "military_vehicles_obj", "military_objects"]:
            src = Path("combined/images/val") / folder
            expected = Path("combined/labels/val") / folder
            self.assertEqual(images_dir_to_labels_dir(src), expected)

    def test_no_images_component_unchanged_except_nothing_to_swap(self):
        # no 'images' component at all -- path passes through unchanged
        p = Path("data/pics/val/foo")
        self.assertEqual(images_dir_to_labels_dir(p), p)


class TestComputeAveragePrecision(unittest.TestCase):
    def test_no_ground_truth_returns_none(self):
        ap, p, r = compute_average_precision(np.array([1.0]), np.array([0.0]), n_gt=0)
        self.assertIsNone(ap)

    def test_perfect_detector(self):
        # 3 predictions, all true positives, exactly 3 ground truths
        tp = np.array([1.0, 1.0, 1.0])
        fp = np.array([0.0, 0.0, 0.0])
        ap, final_p, final_r = compute_average_precision(tp, fp, n_gt=3)
        self.assertAlmostEqual(ap, 1.0, places=5)
        self.assertAlmostEqual(final_p, 1.0, places=5)
        self.assertAlmostEqual(final_r, 1.0, places=5)

    def test_no_predictions_at_all(self):
        # ground truth exists but nothing was predicted
        tp = np.array([])
        fp = np.array([])
        ap, final_p, final_r = compute_average_precision(tp, fp, n_gt=5)
        self.assertAlmostEqual(ap, 0.0, places=5)
        self.assertEqual(final_r, 0.0)

    def test_all_false_positives(self):
        tp = np.array([0.0, 0.0])
        fp = np.array([1.0, 1.0])
        ap, final_p, final_r = compute_average_precision(tp, fp, n_gt=2)
        self.assertAlmostEqual(ap, 0.0, places=5)
        self.assertEqual(final_r, 0.0)

    def test_partial_recall_known_value(self):
        # 1 true positive found out of 2 ground truths -> recall caps at 0.5
        tp = np.array([1.0])
        fp = np.array([0.0])
        ap, final_p, final_r = compute_average_precision(tp, fp, n_gt=2)
        self.assertAlmostEqual(final_r, 0.5, places=5)
        self.assertAlmostEqual(final_p, 1.0, places=5)


class TestComputePrecisionRecallF1(unittest.TestCase):
    def test_perfect_score(self):
        p, r, f1 = compute_precision_recall_f1(tp=10, fp=0, fn=0)
        self.assertEqual((p, r, f1), (1.0, 1.0, 1.0))

    def test_all_zero_is_safe(self):
        # no predictions, no ground truth -- must not divide by zero
        p, r, f1 = compute_precision_recall_f1(tp=0, fp=0, fn=0)
        self.assertEqual((p, r, f1), (0.0, 0.0, 0.0))

    def test_known_values(self):
        # tp=3, fp=1 -> precision = 0.75; tp=3, fn=2 -> recall = 0.6
        p, r, f1 = compute_precision_recall_f1(tp=3, fp=1, fn=2)
        self.assertAlmostEqual(p, 0.75)
        self.assertAlmostEqual(r, 0.6)
        expected_f1 = 2 * 0.75 * 0.6 / (0.75 + 0.6)
        self.assertAlmostEqual(f1, expected_f1)

    def test_no_true_positives(self):
        p, r, f1 = compute_precision_recall_f1(tp=0, fp=5, fn=5)
        self.assertEqual((p, r, f1), (0.0, 0.0, 0.0))


class TestMatchPredictionsToGroundTruth(unittest.TestCase):
    def test_correct_match(self):
        gt = [(0, [0, 0, 10, 10])]  # class 0
        preds = [(0, [0, 0, 10, 10], 0.9)]  # class 0, perfect box
        pairs = match_predictions_to_ground_truth(gt, preds, iou_thresh=0.5)
        self.assertEqual(pairs, [(0, 0)])

    def test_misclassification_still_counted_as_spatial_match(self):
        # spatially overlapping but different classes -> should appear
        # as (gt_class, pred_class) with gt_class != pred_class, not as
        # a separate FP + FN
        gt = [(3, [0, 0, 10, 10])]  # class "tank" (id 3)
        preds = [(7, [0, 0, 10, 10], 0.9)]  # predicted class "drone" (id 7)
        pairs = match_predictions_to_ground_truth(gt, preds, iou_thresh=0.5)
        self.assertEqual(pairs, [(3, 7)])

    def test_false_negative_missed_ground_truth(self):
        gt = [(0, [0, 0, 10, 10])]
        preds = []  # nothing predicted
        pairs = match_predictions_to_ground_truth(gt, preds, iou_thresh=0.5)
        self.assertEqual(pairs, [(0, None)])

    def test_false_positive_spurious_prediction(self):
        gt = []  # nothing in the image
        preds = [(2, [0, 0, 10, 10], 0.9)]
        pairs = match_predictions_to_ground_truth(gt, preds, iou_thresh=0.5)
        self.assertEqual(pairs, [(None, 2)])

    def test_low_iou_treated_as_no_match(self):
        # boxes barely overlap, below threshold -> counted as FP + FN,
        # not as a match
        gt = [(0, [0, 0, 10, 10])]
        preds = [(0, [9, 9, 20, 20], 0.9)]  # tiny overlap
        pairs = match_predictions_to_ground_truth(gt, preds, iou_thresh=0.5)
        self.assertIn((0, None), pairs)
        self.assertIn((None, 0), pairs)
        self.assertEqual(len(pairs), 2)

    def test_higher_confidence_prediction_gets_first_pick(self):
        # two predictions overlap the same gt box; the higher-confidence
        # one (processed first, since preds are pre-sorted by caller)
        # should claim the match, the other becomes a false positive
        gt = [(0, [0, 0, 10, 10])]
        preds = [
            (0, [0, 0, 10, 10], 0.9),   # higher confidence, listed first
            (0, [1, 1, 11, 11], 0.4),   # lower confidence
        ]
        pairs = match_predictions_to_ground_truth(gt, preds, iou_thresh=0.5)
        self.assertEqual(pairs[0], (0, 0))  # first pred matched
        self.assertEqual(pairs[1], (None, 0))  # second pred is a false positive


if __name__ == "__main__":
    unittest.main()
