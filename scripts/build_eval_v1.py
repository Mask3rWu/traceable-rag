"""Build per-document retrieval evaluation candidates from parsed documents.

The output intentionally uses only stable doc.json block IDs.  It is a first
pass over parsed text; visual/PDF validation remains a separate review step.
"""

from __future__ import annotations

import json
import math
import re
from html import unescape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARSED_ROOT = ROOT / "processed" / "parsed"
OUTPUT_ROOT = ROOT / "eval" / "v1"
MIN_QUESTIONS_PER_DOCUMENT = 5

BOILERPLATE = (
    "中华人民共和国",
    "国家军用标准",
    "发布",
    "实施",
    "批准",
    "目次",
    "前言",
    "参考文献",
    "附加说明",
    "印刷",
    "开本",
    "字数",
)

REQUIREMENT_PATTERNS = (
    r"应当",
    r"应(?:按|在|将|由|予以|符合|具有|能|保证|满足|提供|进行|采用|为|包括|控制)",
    r"必须",
    r"不得",
    r"宜(?:按|在|采用|为|具有|保持|设置|选择)",
)


def compact(text: str) -> str:
    text = unescape(re.sub(r"<[^>]+>", " ", text))
    return re.sub(r"\s+", " ", text.replace("#", " ")).strip()


def document_title(data: dict, fallback: str) -> str:
    for page in data["pages"][:3]:
        for block in page["blocks"]:
            if block["block_type"] == "heading":
                title = compact(block.get("text", ""))
                if len(title) >= 6 and not title.startswith(("第", "附录")):
                    return title[:80]
    return fallback.replace("_", " ")


def is_candidate(block: dict) -> bool:
    text = compact(block.get("text", ""))
    if block.get("block_type") not in {"paragraph", "table"} or len(text) < 26:
        return False
    if block.get("confidence", 0) < 0.68:
        return False
    if any(term in text for term in BOILERPLATE):
        return False
    if any(term in text for term in ("University", "Institute", "Centre for", "Correspondence", "Email:")):
        return False
    if re.match(r"^(?:Keywords?|Index Terms?)\s*[:：]", text, re.IGNORECASE):
        return False
    if re.match(r"^[A-Z][A-Z .,-]{12,}(?:\d{2,4})?\s*$", text):
        return False
    if re.match(r"^(?:GJB|GB/?T?|MIL[- ]STD|ISO)\s*[-\d.]", text, re.IGNORECASE):
        return False
    if re.match(r"^《[^》]{4,80}》\s*(?:\d{4}年|\d{4}[-./])", text):
        return False
    if re.match(r"^\[\d+\]", text):
        return False
    if text.startswith(("图", "表", "注：", "注:")) and len(text) < 50:
        return False
    # Formula-only OCR fragments and reference-list entries make poor queries.
    if len(re.sub(r"[^\u4e00-\u9fffA-Za-z]", "", text)) < 14:
        return False
    return True


def is_in_markdown(block: dict, markdown: str) -> bool:
    """Confirm paragraph candidates were read from doc.md before annotation.

    Tables are represented as images in many doc.md files, so their reviewed
    page crop is the readable source instead.
    """
    if block["block_type"] == "table":
        return True
    text = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", compact(block["text"]))
    searchable = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", markdown)
    return len(text) >= 18 and text[:18] in searchable


def evenly_select(candidates: list[dict], count: int) -> list[dict]:
    if len(candidates) <= count:
        return candidates
    selected: list[dict] = []
    seen_ids: set[str] = set()
    for index in range(count):
        position = round(index * (len(candidates) - 1) / (count - 1)) if count > 1 else 0
        block = candidates[position]
        if block["block_id"] not in seen_ids:
            selected.append(block)
            seen_ids.add(block["block_id"])
    # Rounded positions are normally distinct; retain a deterministic fallback.
    for block in candidates:
        if len(selected) == count:
            break
        if block["block_id"] not in seen_ids:
            selected.append(block)
            seen_ids.add(block["block_id"])
    return selected


def subject(text: str) -> str:
    text = compact(text)
    text = re.sub(r"^(?:\d+(?:\.\d+)*[、.]?\s*)", "", text)
    text = re.sub(r"^(?:Abstract|摘要)\s*[:：.]?\s*", "", text, flags=re.IGNORECASE)
    # In prescriptive Chinese prose, the clause before these verbs is generally
    # the object of the requirement and makes a useful, answer-independent topic.
    for marker in ("应当", "不得", "必须", "需要", "采用", "是指"):
        if marker in text:
            head = text.split(marker, 1)[0].rstrip("，。；;：: ")
            if 4 <= len(head) <= 42:
                return head
    for delimiter in ("，", "。", "；", ";", "：", ":"):
        head = text.split(delimiter, 1)[0].strip()
        if 6 <= len(head) <= 42:
            return head
    return text[:36]


def is_requirement(text: str) -> bool:
    return any(re.search(pattern, text) for pattern in REQUIREMENT_PATTERNS)


def has_quantitative_claim(text: str) -> bool:
    """Avoid treating identifiers such as F1 or Sentinel-1 as numeric claims."""
    return bool(
        re.search(r"\d+(?:\.\d+)?\s*(?:%|％|h\b|小时|分钟|秒|年|月|天|米|公里|km\b|倍|个|次|例)", text, re.IGNORECASE)
        or re.search(r"(?:大于|小于|不少于|不超过|≥|≤|>|<)\s*\d", text)
    )


def question_for(block: dict, title: str, ordinal: int, language: str) -> str:
    text = compact(block["text"])
    topic = subject(text)
    is_chinese = language == "zh"
    if block["block_type"] == "table":
        headers = "、".join(re.split(r"\s+", text)[:3])
        if is_chinese:
            return f"表格中关于{headers}列出了哪些对应关系、分类或指标？"
        return f"What relationships, categories, or values does the table show for {headers}?"
    if re.search(r"本(?:标准|规范|文件).{0,8}适用于", text):
        return "该资料规定的适用范围和对象是什么？"
    if "是指" in text:
        return f"文中如何定义{topic}？"
    if is_requirement(text):
        return f"关于{topic}，应当遵循什么要求？"
    if not is_chinese and re.search(r"\b(?:This paper|This study) develops\b", text, re.IGNORECASE):
        return "What method or algorithm does the study develop, and what problem does it address?"
    if not is_chinese and re.search(r"\bF1 score\b", text, re.IGNORECASE):
        return "How does the study describe the F1 score for damage-detection accuracy assessment?"
    if any(marker in text for marker in ("采用", "方法", "模型", "算法", "试验", "验证")):
        if is_chinese:
            return f"针对{topic}，文中采用了什么方法或技术路线？"
        return f"According to the study, what method or approach is reported for {topic}?"
    if has_quantitative_claim(text):
        if is_chinese:
            return f"关于{topic}，文中给出的相关数值、条件或结论是什么？"
        return f"What quantitative condition, result, or conclusion is reported for {topic}?"
    if not is_chinese:
        return f"What does the study report about {topic}?"
    return f"关于{topic}，文中作了怎样的说明？"


def source_granularity(index: int, total: int) -> str:
    # Four prompted-source questions per 20 candidates yields an 80/20 split.
    prompted = {4: "document_approximate", 9: "document_exact", 14: "section_approximate", 19: "section_exact"}
    return prompted.get(index % 20, "none")


def with_source_hint(question: str, granularity: str, title: str, section: list[str], language: str) -> str:
    if language == "en":
        short_title = title.title()[:56]
        if granularity == "document_approximate":
            return f"In the study on {short_title}, {question}"
        if granularity == "document_exact":
            return f"According to \"{title.title()}\", {question}"
        if granularity == "section_approximate":
            return f"In the relevant part of the study, {question}"
        if granularity == "section_exact":
            section_name = ", ".join(section) if section else "the relevant section"
            return f"In section {section_name} of the study, {question}"
        return question
    if granularity == "document_approximate":
        return f"在关于{title[:28]}的资料中，{question}"
    if granularity == "document_exact":
        return f"根据《{title}》，{question}"
    if granularity == "section_approximate":
        return f"在{title[:28]}中有关{section[-1] if section else '相关内容'}的部分，{question}"
    if granularity == "section_exact":
        section_name = "、".join(section) if section else "相关章节"
        return f"在《{title}》的{section_name}，{question}"
    return question


def build_document(path: Path) -> list[dict]:
    data = json.loads((path / "doc.json").read_text(encoding="utf-8"))
    markdown_path = path / "doc.md"
    if not markdown_path.is_file():
        raise ValueError(f"Missing doc.md for {path.name}")
    markdown = markdown_path.read_text(encoding="utf-8")
    title = document_title(data, path.name)
    candidates = [
        block
        for page in data["pages"]
        for block in page["blocks"]
        if is_candidate(block) and is_in_markdown(block, markdown)
    ]
    if not candidates:
        raise ValueError(f"No usable evidence blocks in {path.name}")
    # Match the density of the 13-page reference set (25 questions) and do
    # not cap long documents: a cap would silently leave most pages untested.
    target = max(MIN_QUESTIONS_PER_DOCUMENT, math.ceil(data["total_pages"] * 2))
    selected = evenly_select(candidates, min(target, len(candidates)))
    chinese_characters = sum(len(re.findall(r"[\u4e00-\u9fff]", block["text"])) for block in candidates)
    latin_characters = sum(len(re.findall(r"[A-Za-z]", block["text"])) for block in candidates)
    language = "en" if latin_characters > chinese_characters * 3 else "zh"
    records = []
    for index, block in enumerate(selected):
        granularity = source_granularity(index, len(selected))
        question = with_source_hint(
            question_for(block, title, index, language),
            granularity,
            title,
            block.get("section_path", []),
            language,
        )
        records.append(
            {
                "question_id": f"Q{index + 1:04d}",
                "question": question,
                "evidence_block_ids": [block["block_id"]],
                "source_granularity": granularity,
            }
        )
    return records


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    written = 0
    for path in sorted(PARSED_ROOT.iterdir(), key=lambda item: item.name):
        if not (path / "doc.json").is_file() or path.name == "2-电子系统":
            continue
        records = build_document(path)
        output = OUTPUT_ROOT / f"{path.name}.jsonl"
        output.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )
        written += 1
        print(f"{output.relative_to(ROOT)}: {len(records)}")
    print(f"Generated {written} per-document evaluation sets.")


if __name__ == "__main__":
    main()
