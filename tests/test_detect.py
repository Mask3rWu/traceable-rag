from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.数据处理.detect import detect_pdf


class _Result:
    def __init__(self) -> None:
        self.json = {
            "res": {
                "page_index": 0,
                "width": 100,
                "height": 200,
                "parsing_res_list": [
                    {
                        "block_label": "text",
                        "block_content": "content",
                        "block_bbox": [0, 0, 10, 10],
                    }
                ],
            }
        }

    def save_to_markdown(self, save_path: str) -> None:
        Path(save_path, "page.md").write_text("content", encoding="utf-8")


class _Pipeline:
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, *_args, **_kwargs):
        self.calls += 1
        return iter([_Result()])


class DetectTest(unittest.TestCase):
    def test_predicts_only_once_for_json_and_markdown(self):
        pipe = _Pipeline()
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            with patch("src.数据处理.detect._build_pipeline", return_value=pipe):
                metadata = detect_pdf(Path("sample.pdf"), out_dir)

            self.assertEqual(pipe.calls, 1)
            self.assertTrue(Path(metadata["structurev3_json"]).is_file())
            self.assertTrue(Path(metadata["structurev3_md"]).is_file())
            raw = json.loads((out_dir / "structurev3.json").read_text(encoding="utf-8"))
            self.assertEqual(raw[0]["page_index"], 0)
