from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.data_processing.detect import detect_pdf


class _Result:
    def __init__(
        self,
        page_index: int = 0,
        markdown_name: str = "page.md",
        content: str = "content",
    ) -> None:
        self.markdown_name = markdown_name
        self.content = content
        self.json = {
            "res": {
                "page_index": page_index,
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
        Path(save_path, self.markdown_name).write_text(self.content, encoding="utf-8")


class _Pipeline:
    def __init__(self, results=None) -> None:
        self.calls = 0
        self.results = results or [_Result()]

    def predict(self, *_args, **_kwargs):
        self.calls += 1
        return iter(self.results)


class DetectTest(unittest.TestCase):
    def test_predicts_only_once_for_json_and_markdown(self):
        pipe = _Pipeline()
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            with patch("src.data_processing.detect._build_pipeline", return_value=pipe):
                metadata = detect_pdf(Path("sample.pdf"), out_dir)

            self.assertEqual(pipe.calls, 1)
            self.assertTrue(Path(metadata["structurev3_json"]).is_file())
            self.assertTrue(Path(metadata["structurev3_md"]).is_file())
            raw = json.loads((out_dir / "structurev3.json").read_text(encoding="utf-8"))
            self.assertEqual(raw[0]["page_index"], 0)

    def test_markdown_pages_use_natural_numeric_order(self):
        pipe = _Pipeline([
            _Result(10, "page_10.md", "ten"),
            _Result(2, "page_2.md", "two"),
            _Result(1, "page_1.md", "one"),
        ])
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            with patch("src.data_processing.detect._build_pipeline", return_value=pipe):
                detect_pdf(Path("sample.pdf"), out_dir)

            markdown = (out_dir / "structurev3.md").read_text(encoding="utf-8")
            self.assertEqual(markdown, "one\n\ntwo\n\nten")
            raw = json.loads((out_dir / "structurev3.json").read_text(encoding="utf-8"))
            self.assertEqual([page["page_index"] for page in raw], [1, 2, 10])
