"""Unit tests for turning vision-language model replies into boxes."""

import unittest

from scripts.vlm_predict import parse_reply, rescale_bbox, smart_resize

CLASSES = ["tank", "warship"]


class ParseReplyTests(unittest.TestCase):
    def test_reads_fenced_json_and_keeps_known_labels(self) -> None:
        reply = (
            "```json\n"
            '[{"bbox_2d": [10, 20, 110, 220], "label": "tank"},\n'
            ' {"bbox_2d": [0, 0, 5, 5], "label": "bicycle"}]\n'
            "```"
        )
        self.assertEqual(
            parse_reply(reply, CLASSES), [{"class": "tank", "bbox": [10.0, 20.0, 110.0, 220.0]}]
        )

    def test_empty_list_means_no_boxes(self) -> None:
        self.assertEqual(parse_reply("[]", CLASSES), [])

    def test_unusable_replies_return_none(self) -> None:
        self.assertIsNone(parse_reply("I can see two tanks.", CLASSES))
        self.assertIsNone(parse_reply('[{"bbox_2d": [1, 2', CLASSES))

    def test_skips_malformed_boxes(self) -> None:
        reply = '[{"bbox_2d": [1, 2, 3], "label": "tank"}, {"bbox_2d": "x", "label": "warship"}]'
        self.assertEqual(parse_reply(reply, CLASSES), [])


class ResizeTests(unittest.TestCase):
    def test_small_image_rounds_to_multiples_of_28(self) -> None:
        self.assertEqual(smart_resize(1000, 500), (1008, 504))

    def test_large_image_is_capped_at_max_pixels(self) -> None:
        width, height = smart_resize(4000, 3000)
        self.assertLessEqual(width * height, 28 * 28 * 1280)
        self.assertEqual((width % 28, height % 28), (0, 0))

    def test_rescale_maps_resized_pixels_back(self) -> None:
        self.assertEqual(
            rescale_bbox([0, 0, 504, 252], (1008, 504), (2016, 1008)), [0.0, 0.0, 1008.0, 504.0]
        )


if __name__ == "__main__":
    unittest.main()
