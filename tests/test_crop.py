from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.config import ParseConfig
from src.schema import Block, Page
from src.data_processing.crop import crop_visual_blocks


def make_block(
    block_id: str,
    block_type: str,
    bbox: list[int],
    *,
    caption_of: str | None = None,
) -> Block:
    return Block(
        block_id=block_id,
        document_id="doc",
        page=1,
        block_type=block_type,
        order=1,
        bbox=[value / 100 for value in bbox],
        bbox_pixel=bbox,
        caption_of=caption_of,
    )


class CropTest(unittest.TestCase):
    def test_padding_is_clamped_before_paired_caption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            page_path = root / "page.png"
            Image.new("RGB", (100, 100), "white").save(page_path)
            stale = root / "assets" / "crops" / "stale.png"
            stale.parent.mkdir(parents=True)
            stale.write_bytes(b"stale")

            figure = make_block("doc_P001_B01", "figure", [20, 20, 80, 60])
            caption = make_block(
                "doc_P001_B02",
                "caption",
                [20, 70, 80, 80],
                caption_of=figure.block_id,
            )
            page = Page(
                document_id="doc",
                page=1,
                width=100,
                height=100,
                page_image=str(page_path),
                blocks=[figure, caption],
            )
            config = ParseConfig(
                crop_padding_x_ratio=0.1,
                crop_padding_top_ratio=0.1,
                crop_padding_bottom_ratio=0.5,
                crop_padding_min_px=0,
                crop_caption_gap_px=3,
            )

            count = crop_visual_blocks([page], root, config)

            self.assertEqual(count, 1)
            self.assertFalse(stale.exists())
            self.assertEqual(figure.crop_bbox_pixel, [14, 16, 86, 67])
            crop_path = Path(figure.image_crop)
            self.assertTrue(crop_path.is_file())
            with Image.open(crop_path) as crop:
                self.assertEqual(crop.size, (72, 51))
            self.assertTrue(Path(figure.figure_crop).is_file())

    def test_caption_overlap_is_removed_from_pure_crop(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            page_path = root / "page.png"
            Image.new("RGB", (100, 100), "white").save(page_path)
            figure = make_block("doc_P001_B01", "figure", [10, 10, 90, 80])
            zh = make_block(
                "doc_P001_B02", "caption", [20, 75, 80, 84],
                caption_of=figure.block_id,
            )
            en = make_block(
                "doc_P001_B03", "caption", [15, 86, 85, 95],
                caption_of=figure.block_id,
            )
            page = Page(
                document_id="doc", page=1, width=100, height=100,
                page_image=str(page_path), blocks=[figure, zh, en],
            )
            config = ParseConfig(crop_padding_min_px=2, crop_caption_gap_px=1)

            crop_visual_blocks([page], root, config)

            self.assertEqual(figure.crop_bbox_pixel[3], 74)
            self.assertEqual(figure.figure_crop_bbox_pixel, [8, 8, 92, 97])


if __name__ == "__main__":
    unittest.main()
