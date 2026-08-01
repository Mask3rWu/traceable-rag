"""Run the GJB watermark cleaner and produce pixel-level validation artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import cv2
import fitz
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing.render import render_pdf
from src.data_processing.watermark import prepare_watermark_input


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _masks(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    strong = cv2.inRange(
        hsv,
        np.array([0, 80, 100], dtype=np.uint8),
        np.array([18, 255, 255], dtype=np.uint8),
    ) > 0
    broad = cv2.inRange(
        hsv,
        np.array([0, 8, 120], dtype=np.uint8),
        np.array([20, 255, 255], dtype=np.uint8),
    ) > 0
    return strong, broad


def _page_metrics(before_path: Path, after_path: Path) -> dict:
    before = _read_image(before_path)
    after = _read_image(after_path)
    if before.shape != after.shape:
        raise ValueError(f"Page dimensions differ: {before_path} vs {after_path}")

    strong_before, broad_before = _masks(before)
    strong_after, _ = _masks(after)
    strong_before_count = int(strong_before.sum())
    strong_after_count = int(strong_after.sum())
    removal_rate = (
        1.0 - strong_after_count / strong_before_count
        if strong_before_count
        else 1.0
    )

    pixel_changed = np.any(before != after, axis=2)
    outside_changed = int(np.logical_and(pixel_changed, ~broad_before).sum())
    before_brightness = before.max(axis=2).astype(np.int16)
    after_brightness = after.max(axis=2).astype(np.int16)
    dark_affected = np.logical_and(broad_before, before_brightness < 180)
    if dark_affected.any():
        dark_delta = np.abs(after_brightness - before_brightness)[dark_affected]
        dark_mae = float(dark_delta.mean())
        dark_max_delta = int(dark_delta.max())
    else:
        dark_mae = 0.0
        dark_max_delta = 0

    whitened = np.all(after == 255, axis=2)
    strong_whitened_rate = (
        float(whitened[strong_before].mean()) if strong_before_count else 1.0
    )
    return {
        "width": int(before.shape[1]),
        "height": int(before.shape[0]),
        "strong_orange_pixels_before": strong_before_count,
        "strong_orange_pixels_after": strong_after_count,
        "strong_orange_removal_rate": removal_rate,
        "strong_orange_pixels_whitened_rate": strong_whitened_rate,
        "changed_pixels": int(pixel_changed.sum()),
        "changed_pixels_outside_cleaning_mask": outside_changed,
        "dark_affected_pixels": int(dark_affected.sum()),
        "dark_stroke_brightness_mae": dark_mae,
        "dark_stroke_brightness_max_delta": dark_max_delta,
    }


def _render_page(page: fitz.Page, dpi: int = 100) -> np.ndarray:
    pixmap = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
    return np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height, pixmap.width, pixmap.n
    )


def _validate_pdf(source_path: Path, output_path: Path, detected_pages: set[int]) -> dict:
    source = fitz.open(source_path)
    output = fitz.open(output_path)
    try:
        source_sizes = [
            [float(page.rect.width), float(page.rect.height)] for page in source
        ]
        output_sizes = [
            [float(page.rect.width), float(page.rect.height)] for page in output
        ]
        unchanged_pages = []
        for index in range(min(source.page_count, output.page_count)):
            page_number = index + 1
            if page_number in detected_pages:
                continue
            before = _render_page(source[index])
            after = _render_page(output[index])
            same = before.shape == after.shape and bool(np.array_equal(before, after))
            unchanged_pages.append({"page": page_number, "pixel_exact": same})
        return {
            "source_page_count": source.page_count,
            "output_page_count": output.page_count,
            "page_count_preserved": source.page_count == output.page_count,
            "page_sizes_preserved": source_sizes == output_sizes,
            "output_needs_password": bool(output.needs_pass),
            "undetected_pages": unchanged_pages,
            "undetected_pages_pixel_exact": all(
                item["pixel_exact"] for item in unchanged_pages
            ),
        }
    finally:
        output.close()
        source.close()


def _fit_panel(image: np.ndarray, width: int = 720) -> np.ndarray:
    scale = width / image.shape[1]
    return cv2.resize(
        image, (width, round(image.shape[0] * scale)), interpolation=cv2.INTER_AREA
    )


def _comparison_image(before_path: Path, after_path: Path, page: int) -> np.ndarray:
    before = _fit_panel(_read_image(before_path))
    after = _fit_panel(_read_image(after_path))
    height = max(before.shape[0], after.shape[0])
    canvas = np.full((height + 56, before.shape[1] + after.shape[1], 3), 255, np.uint8)
    canvas[56 : 56 + before.shape[0], : before.shape[1]] = before
    canvas[56 : 56 + after.shape[0], before.shape[1] :] = after
    cv2.putText(canvas, f"Page {page} - BEFORE", (18, 38), 0, 1.0, (0, 0, 0), 2)
    cv2.putText(
        canvas,
        f"Page {page} - AFTER",
        (before.shape[1] + 18, 38),
        0,
        1.0,
        (0, 0, 0),
        2,
    )
    return canvas


def _write_image(path: Path, image: np.ndarray) -> None:
    ok, data = cv2.imencode(".png", image)
    if not ok:
        raise ValueError(f"Cannot encode image: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data.tofile(path)


def _write_visuals(
    page_paths: dict[int, Path], clean_dir: Path, pages: list[int], output_dir: Path
) -> list[str]:
    comparisons = []
    rows = []
    for page in pages:
        comparison = _comparison_image(
            page_paths[page], clean_dir / f"p{page:03d}.png", page
        )
        path = output_dir / f"page_{page:03d}_before_after.png"
        _write_image(path, comparison)
        comparisons.append(str(path))
        rows.append(comparison)
    if rows:
        width = max(row.shape[1] for row in rows)
        padded = []
        for row in rows:
            if row.shape[1] == width:
                padded.append(row)
                continue
            canvas = np.full((row.shape[0], width, 3), 255, np.uint8)
            canvas[:, : row.shape[1]] = row
            padded.append(canvas)
        contact_sheet = output_dir / "detected_pages_before_after.png"
        _write_image(contact_sheet, np.vstack(padded))
        comparisons.append(str(contact_sheet))
    return comparisons


def _report_markdown(report: dict) -> str:
    status = "通过" if report["overall_passed"] else "未通过"
    lines = [
        "# 水印处理验证报告",
        "",
        f"- 总体结论：**{status}**",
        f"- 源文件：`{report['source']['path']}`",
        f"- 清洗后 PDF：`{report['output']['path']}`",
        f"- 检测页：{', '.join(map(str, report['detected_pages']))}",
        f"- 页数：{report['pdf_integrity']['source_page_count']} -> {report['pdf_integrity']['output_page_count']}",
        "",
        "## 验证项",
        "",
        "| 验证项 | 结果 |",
        "| --- | --- |",
    ]
    for name, passed in report["checks"].items():
        lines.append(f"| {name} | {'通过' if passed else '未通过'} |")
    lines.extend(
        [
            "",
            "## 命中页指标",
            "",
            "| 页码 | 强橙色去除率 | 强橙色转白率 | 暗色笔画亮度 MAE | 清洗掩膜外变化像素 |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in report["page_metrics"]:
        lines.append(
            "| {page} | {strong_orange_removal_rate:.4%} | "
            "{strong_orange_pixels_whitened_rate:.4%} | "
            "{dark_stroke_brightness_mae:.4f} | "
            "{changed_pixels_outside_cleaning_mask} |".format(**item)
        )
    lines.extend(
        [
            "",
            "说明：暗色笔画指标比较清洗前后的最大通道亮度，用于验证叠在水印上的黑色文字笔画未被漂白。",
            "未命中页通过 100 DPI 渲染逐像素比较，确认处理 PDF 中这些页面未发生变化。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    pdf_path = args.pdf.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_hash_before = _sha256(pdf_path)
    rendered_pages = render_pdf(pdf_path, output_dir, dpi=args.dpi)
    prediction_pdf, detections, metadata_path = prepare_watermark_input(
        pdf_path, output_dir, rendered_pages
    )
    if prediction_pdf == pdf_path:
        raise RuntimeError("No repeated orange GJB watermark was detected")

    clean_pdf = output_dir / "watermark_cleaned.pdf"
    shutil.copy2(prediction_pdf, clean_pdf)
    page_paths = {
        int(item["page"]): Path(item["page_image"]).resolve()
        for item in rendered_pages
    }
    detected_pages = sorted(detections)
    metrics = []
    for page in detected_pages:
        item = _page_metrics(
            page_paths[page], output_dir / "pages_clean" / f"p{page:03d}.png"
        )
        item["page"] = page
        item["detection"] = asdict(detections[page])
        metrics.append(item)

    comparisons = _write_visuals(
        page_paths, output_dir / "pages_clean", detected_pages, output_dir / "comparisons"
    )
    integrity = _validate_pdf(pdf_path, clean_pdf, set(detected_pages))
    source_hash_after = _sha256(pdf_path)
    checks = {
        "检测到重复模板水印": len(detected_pages) >= 2,
        "强橙色像素去除率不低于 99.9%": all(
            item["strong_orange_removal_rate"] >= 0.999 for item in metrics
        ),
        "强橙色像素转白率不低于 99.9%": all(
            item["strong_orange_pixels_whitened_rate"] >= 0.999 for item in metrics
        ),
        "暗色文字笔画亮度保持不变": all(
            item["dark_stroke_brightness_max_delta"] == 0 for item in metrics
        ),
        "清洗掩膜外像素保持不变": all(
            item["changed_pixels_outside_cleaning_mask"] == 0 for item in metrics
        ),
        "PDF 页数和页面尺寸保持不变": bool(
            integrity["page_count_preserved"] and integrity["page_sizes_preserved"]
        ),
        "未命中页面逐像素保持不变": bool(
            integrity["undetected_pages_pixel_exact"]
        ),
        "源 PDF 未被修改": source_hash_before == source_hash_after,
    }
    report = {
        "overall_passed": all(checks.values()),
        "source": {
            "path": str(pdf_path),
            "size_bytes": pdf_path.stat().st_size,
            "sha256_before": source_hash_before,
            "sha256_after": source_hash_after,
        },
        "output": {
            "path": str(clean_pdf),
            "size_bytes": clean_pdf.stat().st_size,
            "sha256": _sha256(clean_pdf),
            "watermark_metadata": str(metadata_path),
            "comparisons": comparisons,
        },
        "detected_pages": detected_pages,
        "checks": checks,
        "page_metrics": metrics,
        "pdf_integrity": integrity,
    }
    (output_dir / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "VALIDATION.md").write_text(
        _report_markdown(report), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
