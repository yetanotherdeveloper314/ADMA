"""Unit tests for the open-vocabulary detector helpers (no models are loaded)."""

import unittest

from scripts.open_vocab_predict import build_prompts, latency_stats, owlv2_to_boxes

CLASSES = ["military_truck", "tank"]


class PromptTests(unittest.TestCase):
    def test_defaults_to_class_names_with_spaces(self) -> None:
        self.assertEqual(build_prompts(CLASSES), ["military truck", "tank"])

    def test_overrides_replace_only_named_classes(self) -> None:
        prompts = build_prompts(CLASSES, {"military_truck": "army truck"})
        self.assertEqual(prompts, ["army truck", "tank"])

    def test_unknown_class_in_overrides_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            build_prompts(CLASSES, {"submarine": "submarine"})


class Owlv2ConversionTests(unittest.TestCase):
    def test_maps_prompt_labels_back_to_class_names(self) -> None:
        raw = [
            {
                "score": 0.42,
                "label": "army truck",
                "box": {"xmin": 1, "ymin": 2, "xmax": 30, "ymax": 40},
            }
        ]
        boxes = owlv2_to_boxes(raw, ["army truck", "tank"], CLASSES)
        self.assertEqual(
            boxes, [{"class": "military_truck", "confidence": 0.42, "bbox": [1, 2, 30, 40]}]
        )


class LatencyTests(unittest.TestCase):
    def test_reports_mean_median_and_p95(self) -> None:
        stats = latency_stats([float(v) for v in range(1, 101)])
        self.assertEqual(stats["images"], 100)
        self.assertAlmostEqual(stats["mean_ms"], 50.5)
        self.assertAlmostEqual(stats["median_ms"], 50.5)
        self.assertGreater(stats["p95_ms"], 94)


if __name__ == "__main__":
    unittest.main()
