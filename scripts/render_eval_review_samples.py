"""Render selected v1 evidence blocks from original page images for visual review."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
PARSED_ROOT = ROOT / "processed" / "parsed"
EVAL_ROOT = ROOT / "eval" / "v1"
OUTPUT_ROOT = ROOT / "_inspect" / "eval_review_samples"
SAMPLES = {
    "2-电子系统": ["Q0005", "Q0015", "Q0018", "Q0025"],
    "无人机通用规范": ["Q0001", "Q0005", "Q0012", "Q0020"],
    "侯鹏_等_-_2025_-_基于毁伤评估结果的无人机对地攻击任务分配方法": ["Q0001", "Q0007", "Q0014", "Q0020"],
    "Ballinger_-_2024_-_Open_access_battle_damage_detection_via_pixel-wise_T-test_on_sentinel-1_imagery": [
        "Q0002", "Q0007", "Q0016", "Q0025"
    ],
    "GJB_Z_1391-2006_故障模式_影响及危害性分析指南": ["Q0012", "Q0022", "Q0036", "Q0043"],
}


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for document_id, question_ids in SAMPLES.items():
        document = json.loads((PARSED_ROOT / document_id / "doc.json").read_text(encoding="utf-8"))
        block_index = {
            block["block_id"]: (page, block)
            for page in document["pages"]
            for block in page["blocks"]
        }
        rows = {
            row["question_id"]: row
            for row in (
                json.loads(line)
                for line in (EVAL_ROOT / f"{document_id}.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        }
        for question_id in question_ids:
            row = rows[question_id]
            block_id = row["evidence_block_ids"][0]
            page, block = block_index[block_id]
            with Image.open(ROOT / page["page_image"]) as source:
                left, top, right, bottom = block["bbox_pixel"]
                padding = 30
                crop = source.crop((max(0, left - padding), max(0, top - padding), min(page["width"], right + padding), min(page["height"], bottom + padding))).convert("RGB")
            # Preserve original pixels; only add a compact reviewer label below.
            label = f"{question_id} | {block_id} | p{page['page']}"
            canvas = Image.new("RGB", (crop.width, crop.height + 36), "white")
            canvas.paste(crop, (0, 0))
            ImageDraw.Draw(canvas).text((8, crop.height + 8), label, fill="black")
            canvas.save(OUTPUT_ROOT / f"{document_id}__{question_id}.png")


if __name__ == "__main__":
    main()
