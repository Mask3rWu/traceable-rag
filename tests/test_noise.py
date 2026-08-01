from __future__ import annotations

import unittest

from src.data_processing.noise import filter_document_noise
from src.schema import Block, Document, Page


def make_block(
    block_id: str,
    text: str,
    *,
    page: int,
    block_type: str = "paragraph",
    raw_label: str = "text",
    confidence: float = 0.9,
    bbox: list[float] | None = None,
) -> Block:
    return Block(
        block_id=block_id,
        document_id="doc",
        page=page,
        block_type=block_type,
        bbox=bbox or [0.1, 0.2, 0.9, 0.3],
        bbox_pixel=[10, 20, 90, 30],
        text=text,
        confidence=confidence,
        raw_label=raw_label,
    )


def make_page(page: int, blocks: list[Block]) -> Page:
    return Page(document_id="doc", page=page, width=100, height=100, blocks=blocks)


class NoiseFilterTest(unittest.TestCase):
    def test_removes_each_deterministic_noise_category(self):
        document = Document(
            document_id="doc",
            source_file="doc.pdf",
            total_pages=5,
            pages=[
                make_page(
                    1,
                    [
                        make_block("toc", "1 Scope", page=1, raw_label="content"),
                        make_block("foot", "Received: 2025-01-01", page=1, block_type="footnote"),
                        make_block("doi", "DOI: 10.1/example", page=1),
                        make_block("symbol", "（）", page=1, block_type="caption"),
                        make_block("figure-label", "a)", page=1, block_type="caption"),
                    ],
                ),
                make_page(
                    2,
                    [
                        make_block(
                            "header-2",
                            "Journal Name",
                            page=2,
                            bbox=[0.08, 0.875, 0.28, 0.89],
                        )
                    ],
                ),
                make_page(
                    3,
                    [
                        make_block(
                            "header-3",
                            "Journal Name",
                            page=3,
                            bbox=[0.72, 0.875, 0.92, 0.89],
                        )
                    ],
                ),
                make_page(
                    4,
                    [
                        make_block(
                            "header-4",
                            "Journal Name",
                            page=4,
                            bbox=[0.08, 0.876, 0.28, 0.891],
                        )
                    ],
                ),
                make_page(5, [make_block("artifact", "SAG", page=5, confidence=0.0)]),
            ],
        )

        counts = filter_document_noise(document)

        self.assertEqual(
            counts,
            {
                "table_of_contents": 1,
                "footnote_metadata": 1,
                "running_header": 3,
                "inline_publication_metadata": 1,
                "symbol_fragment": 1,
                "isolated_unscored_fragment": 1,
            },
        )
        remaining = [block.block_id for page in document.pages for block in page.blocks]
        self.assertEqual(remaining, ["figure-label"])

    def test_repeated_body_text_is_not_a_running_header(self):
        document = Document(
            document_id="doc",
            source_file="doc.pdf",
            total_pages=3,
            pages=[
                make_page(
                    page,
                    [
                        make_block(
                            f"body-{page}",
                            "Implementation steps are as follows.",
                            page=page,
                            bbox=[0.15, 0.87, 0.45, 0.89],
                        )
                    ],
                )
                for page in range(1, 4)
            ],
        )

        counts = filter_document_noise(document)

        self.assertEqual(counts["running_header"], 0)
        self.assertEqual(
            [block.block_id for page in document.pages for block in page.blocks],
            ["body-1", "body-2", "body-3"],
        )
