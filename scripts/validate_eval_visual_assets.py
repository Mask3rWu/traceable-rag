"""Check that every v1 evidence block remains visually reviewable.

This does not claim that OCR text is correct.  It verifies the prerequisite
for that review: each annotation resolves to an original rendered page and a
valid block bounding box that can be cropped from that page.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
PARSED_ROOT = ROOT / "processed" / "parsed"
EVAL_ROOT = ROOT / "eval" / "v1"
SOURCE_ROOT = ROOT / "资料"


def main() -> None:
    checked = 0
    nonempty_crops = 0
    issues: list[str] = []
    documents: set[str] = set()
    source_pdfs = {path.name for path in SOURCE_ROOT.rglob("*.pdf")}
    page_cache: OrderedDict[Path, Image.Image] = OrderedDict()
    for eval_file in sorted(EVAL_ROOT.glob("*.jsonl"), key=lambda item: item.name):
        source_file = PARSED_ROOT / eval_file.stem / "doc.json"
        if not source_file.exists():
            continue
        document = json.loads(source_file.read_text(encoding="utf-8"))
        if document["source_file"] not in source_pdfs:
            issues.append(f"{eval_file.name}: original PDF missing: {document['source_file']}")
        blocks = {
            block["block_id"]: (page, block)
            for page in document["pages"]
            for block in page["blocks"]
        }
        for line_number, line in enumerate(eval_file.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            for block_id in row["evidence_block_ids"]:
                checked += 1
                page, block = blocks[block_id]
                image = ROOT / page["page_image"]
                bbox = block.get("bbox_pixel")
                if not image.is_file():
                    issues.append(f"{eval_file.name}:{line_number}: page image missing: {image}")
                    continue
                if not isinstance(bbox, list) or len(bbox) != 4:
                    issues.append(f"{eval_file.name}:{line_number}: invalid bbox for {block_id}")
                    continue
                left, top, right, bottom = bbox
                if not (0 <= left < right <= page["width"] and 0 <= top < bottom <= page["height"]):
                    issues.append(f"{eval_file.name}:{line_number}: bbox outside page for {block_id}")
                    continue
                if image not in page_cache:
                    with Image.open(image) as page_image:
                        page_cache[image] = page_image.convert("L")
                    if len(page_cache) > 8:
                        page_cache.popitem(last=False)
                else:
                    page_cache.move_to_end(image)
                crop = page_cache[image].crop((left, top, right, bottom))
                # A valid evidence region should contain foreground pixels.
                # Thresholding avoids treating a nearly white blank crop as text.
                foreground = sum(count for value, count in enumerate(crop.histogram()) if value < 245)
                if foreground / (crop.width * crop.height) < 0.0005:
                    issues.append(f"{eval_file.name}:{line_number}: visually blank crop for {block_id}")
                else:
                    nonempty_crops += 1
                documents.add(eval_file.stem)

    print(f"Documents with visual evidence: {len(documents)}")
    print(f"Evidence references checked for page/bbox: {checked}/{checked}")
    print(f"Evidence crops with visible foreground: {nonempty_crops}/{checked}")
    if issues:
        print("Visual-asset validation failed:")
        print("\n".join(issues))
        raise SystemExit(1)
    print("Visual-asset validation passed.")


if __name__ == "__main__":
    main()
