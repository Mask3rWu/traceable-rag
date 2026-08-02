"""Validate v1 evaluation JSONL files against their parsed source documents."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARSED_ROOT = ROOT / "processed" / "parsed"
EVAL_ROOT = ROOT / "eval" / "v1"
REQUIRED_FIELDS = {"question_id", "question", "evidence_block_ids", "source_granularity"}
VALID_GRANULARITIES = {
    "none",
    "document_approximate",
    "document_exact",
    "section_approximate",
    "section_exact",
}


def main() -> None:
    documents = {path.name for path in PARSED_ROOT.iterdir() if (path / "doc.json").is_file()}
    outputs = {path.stem for path in EVAL_ROOT.glob("*.jsonl")}
    errors: list[str] = []
    if missing := documents - outputs:
        errors.append(f"Missing test sets: {sorted(missing)}")
    if orphaned := outputs - documents:
        errors.append(f"Test sets without documents: {sorted(orphaned)}")

    total = 0
    granularities: Counter[str] = Counter()
    page_coverage: dict[str, tuple[int, int]] = {}
    for filename in sorted(EVAL_ROOT.glob("*.jsonl"), key=lambda item: item.name):
        source = PARSED_ROOT / filename.stem / "doc.json"
        if not source.is_file():
            continue
        data = json.loads(source.read_text(encoding="utf-8"))
        valid_ids = {block["block_id"] for page in data["pages"] for block in page["blocks"]}
        evidence_pages: set[int] = set()
        rows = [json.loads(line) for line in filename.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows:
            errors.append(f"{filename.name}: empty")
            continue
        for index, row in enumerate(rows, start=1):
            total += 1
            if set(row) != REQUIRED_FIELDS:
                errors.append(f"{filename.name}:{index}: fields must be {sorted(REQUIRED_FIELDS)}")
                continue
            granularities[row["source_granularity"]] += 1
            if row["source_granularity"] not in VALID_GRANULARITIES:
                errors.append(f"{filename.name}:{index}: invalid source_granularity")
            if not isinstance(row["question"], str) or not row["question"].strip():
                errors.append(f"{filename.name}:{index}: blank question")
            if not isinstance(row["evidence_block_ids"], list) or not row["evidence_block_ids"]:
                errors.append(f"{filename.name}:{index}: evidence_block_ids must be nonempty list")
            elif any(block_id not in valid_ids for block_id in row["evidence_block_ids"]):
                errors.append(f"{filename.name}:{index}: unknown evidence block")
            else:
                block_pages = {
                    block["block_id"]: page["page"]
                    for page in data["pages"]
                    for block in page["blocks"]
                }
                evidence_pages.update(block_pages[block_id] for block_id in row["evidence_block_ids"])
            if row["question_id"] != f"Q{index:04d}":
                errors.append(f"{filename.name}:{index}: question IDs must be sequential")
        page_coverage[filename.stem] = (len(evidence_pages), data["total_pages"])

    prompted_ratio = (total - granularities["none"]) / total if total else 0
    print(f"Documents: {len(documents)}; test sets: {len(outputs)}; questions: {total}")
    print(f"Source granularity: {dict(sorted(granularities.items()))}")
    print(f"Prompted-source ratio: {prompted_ratio:.1%}")
    lowest_coverage = sorted(page_coverage.items(), key=lambda item: item[1][0] / item[1][1])[:5]
    print("Lowest page coverage: " + ", ".join(f"{name} {covered}/{pages}" for name, (covered, pages) in lowest_coverage))
    if errors:
        print("Validation failed:")
        print("\n".join(errors))
        raise SystemExit(1)
    print("Validation passed.")


if __name__ == "__main__":
    main()
