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
                        "block_content": content,
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
        self.inputs = []
        self.results = results or [_Result()]

    def predict(self, *args, **_kwargs):
        self.calls += 1
        self.inputs.append(args[0])
        return iter(self.results)


class DetectTest(unittest.TestCase):
    def test_predicts_only_once_for_json_and_markdown(self):
        pipe = _Pipeline()
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            with patch("src.data_processing.detect._build_pipeline", return_value=pipe):
                metadata = detect_pdf(Path("sample.pdf"), out_dir)

            self.assertEqual(pipe.calls, 1)
            self.assertTrue(Path(metadata["structure_json"]).is_file())
            self.assertTrue(Path(metadata["structure_md"]).is_file())
            raw = json.loads((out_dir / "structure.json").read_text(encoding="utf-8"))
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

            markdown = (out_dir / "structure.md").read_text(encoding="utf-8")
            self.assertEqual(markdown, "one\n\ntwo\n\nten")
            raw = json.loads((out_dir / "structure.json").read_text(encoding="utf-8"))
            self.assertEqual([page["page_index"] for page in raw], [1, 2, 10])

    def test_no_watermark_keeps_original_prediction_input(self):
        import cv2
        import numpy as np

        pipe = _Pipeline()
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            pages = []
            for page_num in (1, 2):
                image_path = out_dir / f"p{page_num:03d}.png"
                image = np.full((200, 100, 3), 255, dtype=np.uint8)
                ok, encoded = cv2.imencode(".png", image)
                self.assertTrue(ok)
                encoded.tofile(image_path)
                pages.append({"page": page_num, "page_image": str(image_path)})

            with patch("src.data_processing.detect._build_pipeline", return_value=pipe):
                detect_pdf(Path("sample.pdf"), out_dir, rendered_pages=pages)

            self.assertEqual(pipe.inputs, ["sample.pdf"])
            self.assertFalse((out_dir / "_watermark_cleaned_input.pdf").exists())

    def test_watermark_prediction_replaces_only_detected_pages(self):
        from src.data_processing.watermark import WatermarkDetection

        class SplitPipeline(_Pipeline):
            def predict(self, input_path, **_kwargs):
                self.calls += 1
                self.inputs.append(input_path)
                prefix = "clean" if input_path.endswith("cleaned.pdf") else "original"
                return iter([
                    _Result(0, content=f"{prefix}-0"),
                    _Result(1, content=f"{prefix}-1"),
                ])

        pipe = SplitPipeline()
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            cleaned_pdf = out_dir / "cleaned.pdf"
            cleaned_pdf.write_bytes(b"temporary")
            metadata = out_dir / "watermarks.json"
            metadata.write_text("{}", encoding="utf-8")
            detection = WatermarkDetection(
                page=2,
                watermark_type="orange_gjb",
                mask_ratio=0.07,
                largest_component_ratio=0.04,
                bbox=[0.2, 0.2, 0.8, 0.7],
                template_similarity=0.7,
            )

            with (
                patch("src.data_processing.detect._build_pipeline", return_value=pipe),
                patch(
                    "src.data_processing.watermark.prepare_watermark_input",
                    return_value=(cleaned_pdf, {2: detection}, metadata),
                ),
            ):
                detect_pdf(
                    Path("sample.pdf"),
                    out_dir,
                    rendered_pages=[{"page": 1}, {"page": 2}],
                )

            raw = json.loads((out_dir / "structure.json").read_text("utf-8"))
            contents = [page["parsing_res_list"][0]["block_content"] for page in raw]
            self.assertEqual(contents, ["original-0", "clean-1"])
            self.assertEqual(pipe.inputs, ["sample.pdf", str(cleaned_pdf)])
            self.assertFalse(cleaned_pdf.exists())
