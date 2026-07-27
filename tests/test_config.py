from __future__ import annotations

import unittest

from src.config import ParseConfig


class ConfigTest(unittest.TestCase):
    def test_mobile_models_are_explicit(self):
        kwargs = ParseConfig().to_pipeline_kwargs()
        self.assertEqual(kwargs["layout_detection_model_name"], "PP-DocLayoutV3")
        self.assertEqual(kwargs["text_detection_model_name"], "PP-OCRv5_mobile_det")
        self.assertEqual(kwargs["text_recognition_model_name"], "PP-OCRv5_mobile_rec")
        self.assertEqual(kwargs["formula_recognition_model_name"], "PP-FormulaNet_plus-M")
