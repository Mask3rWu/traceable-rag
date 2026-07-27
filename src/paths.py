"""路径常量。

所有路径相对项目根解析，避免依赖 cwd。代码与数据分离：
资料/ 只读，processed/ 是机器产物。
"""
from __future__ import annotations

from pathlib import Path

# 项目根：src/ 的上一级
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 原始资料（只读，永不修改）
DATA_ROOT = PROJECT_ROOT / "资料"
GJB_DIR = DATA_ROOT / "国军标"
PAPER_DIR = DATA_ROOT / "论文"

# 机器处理产物
PROCESSED_ROOT = PROJECT_ROOT / "processed"
PARSED_ROOT = PROCESSED_ROOT / "parsed"  # 解析层输出


def doc_out_dir(doc_id: str) -> Path:
    """单文档解析输出目录，并保证 pages/ 与 assets/ 子目录存在。"""
    d = PARSED_ROOT / doc_id
    (d / "pages").mkdir(parents=True, exist_ok=True)
    (d / "assets").mkdir(parents=True, exist_ok=True)
    return d


def list_input_pdfs() -> list[Path]:
    """枚举所有待解析 PDF（国军标 + 论文）。"""
    pdfs: list[Path] = []
    for src in (GJB_DIR, PAPER_DIR):
        if src.exists():
            pdfs.extend(sorted(src.glob("*.pdf")))
    return pdfs


def doc_id_from_path(pdf_path: Path) -> str:
    """doc_id 规则：文件名去后缀，空格/特殊字符转下划线。"""
    import re

    name = pdf_path.stem
    name = re.sub(r"[^\w一-鿿\-]", "_", name)  # 保留中文/字母/数字/下划线/连字符
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "doc"
