"""分析刚跑的国军标解析结果，与论文基线对比，判断是否需要独立流程。"""
from __future__ import annotations
import json, sys
from collections import Counter
from pathlib import Path

GJB_IDS = [
    "7-装甲车辆和军用汽车",
    "建（构）筑物地震破坏等级划分",
    "无人机通用规范",
]

# 论文基线（来自上一步实测）
PAPER_BASELINE = {
    "caption配对率": "62%~87%",
    "references/块": "~0.1~0.15",
    "heading最深层级": "3",
    "source_method": "全部 ocr（论文也有文本层但 §6.5 未实现）",
}


def analyze(doc_id: str) -> dict | None:
    path = Path("processed/parsed") / doc_id / "doc.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    blocks = [b for pg in data["pages"] for b in pg["blocks"]]
    if not blocks:
        return {"doc_id": doc_id, "empty": True}
    c = Counter(b["block_type"] for b in blocks)
    sm = Counter(b.get("source_method", "?") for b in blocks)
    # heading 层级
    sp = [len(b["section_path"]) for b in blocks if b["block_type"] == "heading" and b.get("section_path")]
    sp_dist = Counter(sp)
    # caption 配对率
    caps = [b for b in blocks if b["block_type"] == "caption"]
    paired = [b for b in caps if b.get("caption_of")]
    figs = [b for b in blocks if b["block_type"] in ("figure", "table")]
    refs = sum(len(b.get("references") or []) for b in blocks)
    # OCR 质量采样：取若干 paragraph 看平均文本长度与可疑字符比例
    paras = [b for b in blocks if b["block_type"] == "paragraph"]
    total_chars = sum(len(b.get("text", "")) for b in paras)
    avg_len = total_chars / len(paras) if paras else 0
    # 空文本块 / 极短文本块比例
    empty_or_short = sum(1 for b in paras if len(b.get("text", "").strip()) < 3)
    return {
        "doc_id": doc_id,
        "pages": data["total_pages"],
        "blocks": len(blocks),
        "block_type": dict(c),
        "source_method": dict(sm),
        "heading层级(深:数)": dict(sp_dist),
        f"caption配对: {len(paired)}/{len(caps)}": f"图表块{len(figs)}个",
        "references总数": refs,
        "paragraph平均字符": round(avg_len, 1),
        "极短/空paragraph": f"{empty_or_short}/{len(paras)}",
    }


if __name__ == "__main__":
    print("=== 论文基线 ===")
    for k, v in PAPER_BASELINE.items():
        print(f"  {k}: {v}")
    print("\n=== 国军标解析结果 ===")
    for gid in GJB_IDS:
        r = analyze(gid)
        if r is None:
            print(f"\n[{gid}] 未找到 doc.json（解析失败或未完成）")
            continue
        if r.get("empty"):
            print(f"\n[{gid}] 空结果")
            continue
        print(f"\n[{r.pop('doc_id')}] ({r.pop('pages')}p, {r.pop('blocks')}块)")
        for k, v in r.items():
            print(f"  {k}: {v}")
