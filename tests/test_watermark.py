from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.data_processing.watermark import (
    clean_orange_watermark,
    detect_orange_gjb_watermarks,
    prepare_watermark_input,
)


def _write_page(path: Path, *, large_center_mark: bool) -> None:
    image = np.full((800, 600, 3), 255, dtype=np.uint8)
    if large_center_mark:
        cv2.rectangle(image, (120, 240), (480, 560), (60, 110, 245), -1)
        cv2.line(image, (150, 400), (450, 400), (0, 0, 0), 5)
    else:
        # Ordinary small colored content must not activate the document profile.
        cv2.rectangle(image, (20, 20), (70, 70), (60, 110, 245), -1)
        cv2.putText(image, "body", (120, 300), 0, 1, (0, 0, 0), 2)
    self_encoded, data = cv2.imencode(".png", image)
    if not self_encoded:
        raise RuntimeError("failed to encode fixture")
    data.tofile(path)


class WatermarkTest(unittest.TestCase):
    def test_repeated_large_center_marks_are_detected_and_cleaned(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "p001.png"
            second = root / "p002.png"
            _write_page(first, large_center_mark=True)
            _write_page(second, large_center_mark=True)

            detections = detect_orange_gjb_watermarks([(1, first), (2, second)])
            self.assertEqual([item.page for item in detections], [1, 2])
            self.assertTrue(all(item.template_similarity >= 0.55 for item in detections))

            cleaned = root / "cleaned.png"
            clean_orange_watermark(first, cleaned)
            image = cv2.imdecode(
                np.fromfile(cleaned, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            self.assertTrue(np.all(image[300, 300] == 255))
            self.assertTrue(np.all(image[400, 300] == 0))

    def test_small_or_single_orange_content_does_not_activate_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            small_a = root / "small_a.png"
            small_b = root / "small_b.png"
            large = root / "large.png"
            _write_page(small_a, large_center_mark=False)
            _write_page(small_b, large_center_mark=False)
            _write_page(large, large_center_mark=True)

            self.assertEqual(
                detect_orange_gjb_watermarks([(1, small_a), (2, small_b)]), []
            )
            self.assertEqual(detect_orange_gjb_watermarks([(1, large)]), [])

    def test_no_match_returns_original_pdf_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.pdf"
            source.write_bytes(b"not opened when no watermark is detected")
            first = root / "p001.png"
            second = root / "p002.png"
            _write_page(first, large_center_mark=False)
            _write_page(second, large_center_mark=False)
            rendered = [
                {"page": 1, "page_image": str(first)},
                {"page": 2, "page_image": str(second)},
            ]

            prediction_input, detections, metadata = prepare_watermark_input(
                source, root / "out", rendered
            )

            self.assertEqual(prediction_input, source)
            self.assertEqual(detections, {})
            data = json.loads(metadata.read_text(encoding="utf-8"))
            self.assertEqual(data["detected_pages"], [])
            self.assertFalse((root / "out" / "_watermark_cleaned_input.pdf").exists())


if __name__ == "__main__":
    unittest.main()
