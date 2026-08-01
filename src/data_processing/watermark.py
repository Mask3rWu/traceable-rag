"""Targeted detection and suppression for the repeated orange GJB watermark."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import cv2
import fitz
import numpy as np


PROFILE_NAME = "orange_gjb"


@dataclass
class WatermarkDetection:
    page: int
    watermark_type: str
    mask_ratio: float
    largest_component_ratio: float
    bbox: list[float]
    template_similarity: float
    cleaned_image: str | None = None


def _load_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取水印检测页图: {path}")
    return image


def _write_png(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError(f"无法编码清洗页图: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(path)


def _strong_orange_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return cv2.inRange(
        hsv,
        np.array([0, 80, 100], dtype=np.uint8),
        np.array([18, 255, 255], dtype=np.uint8),
    )


def _candidate(page: int, image_path: Path) -> tuple[WatermarkDetection, np.ndarray] | None:
    image = _load_image(image_path)
    mask = _strong_orange_mask(image)
    height, width = mask.shape
    page_area = float(width * height)
    mask_ratio = float(np.count_nonzero(mask) / page_area)
    if mask_ratio < 0.05:
        return None

    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return None
    component_index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, box_width, box_height, area = stats[component_index]
    component_ratio = float(area / page_area)
    relative_width = box_width / width
    relative_height = box_height / height
    center_x = (x + box_width / 2) / width
    center_y = (y + box_height / 2) / height

    # This profile is intentionally strict. It targets the large, central GJB mark
    # and leaves ordinary orange charts, highlights and seals alone.
    if not (
        component_ratio >= 0.03
        and 0.45 <= relative_width <= 0.75
        and 0.28 <= relative_height <= 0.52
        and 0.35 <= center_x <= 0.65
        and 0.38 <= center_y <= 0.62
    ):
        return None

    mask_y, mask_x = np.where(mask > 0)
    mask_left = int(mask_x.min())
    mask_top = int(mask_y.min())
    mask_right = int(mask_x.max()) + 1
    mask_bottom = int(mask_y.max()) + 1

    detection = WatermarkDetection(
        page=page,
        watermark_type=PROFILE_NAME,
        mask_ratio=mask_ratio,
        largest_component_ratio=component_ratio,
        bbox=[
            float(mask_left / width),
            float(mask_top / height),
            float(mask_right / width),
            float(mask_bottom / height),
        ],
        template_similarity=0.0,
    )
    normalized_mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST) > 0
    return detection, normalized_mask


def detect_orange_gjb_watermarks(
    page_images: Sequence[tuple[int, Path]],
) -> list[WatermarkDetection]:
    """Return only repeated, template-like orange GJB candidates.

    Requiring a second similar page is the main false-positive guard for unrelated
    PDFs containing a large orange chart or illustration.
    """
    candidates: list[tuple[WatermarkDetection, np.ndarray]] = []
    for page, image_path in page_images:
        found = _candidate(page, image_path)
        if found is not None:
            candidates.append(found)
    if len(candidates) < 2:
        return []

    confirmed: list[WatermarkDetection] = []
    for index, (detection, mask) in enumerate(candidates):
        best_similarity = 0.0
        for other_index, (_, other_mask) in enumerate(candidates):
            if index == other_index:
                continue
            union = np.logical_or(mask, other_mask).sum()
            if union:
                similarity = float(np.logical_and(mask, other_mask).sum() / union)
                best_similarity = max(best_similarity, similarity)
        if best_similarity >= 0.55:
            detection.template_similarity = best_similarity
            confirmed.append(detection)
    return confirmed


def clean_orange_watermark(image_path: Path, output_path: Path) -> None:
    """Neutralize orange pixels while retaining dark overprinted text strokes."""
    image = _load_image(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    broad_mask = cv2.inRange(
        hsv,
        np.array([0, 8, 120], dtype=np.uint8),
        np.array([20, 255, 255], dtype=np.uint8),
    ) > 0

    brightest = image.max(axis=2)
    neutral = brightest.copy()
    # The watermark itself is bright. Darker mixed pixels may carry black glyph
    # strokes, so keep their intensity instead of blindly painting them white.
    neutral[(broad_mask) & (brightest >= 180)] = 255
    image[broad_mask] = np.repeat(neutral[:, :, None], 3, axis=2)[broad_mask]
    _write_png(output_path, image)


def _resolve_page_image(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    from src.paths import PROJECT_ROOT

    return PROJECT_ROOT / path


def _stored_path(path: Path) -> str:
    from src.paths import PROJECT_ROOT

    try:
        value = path.resolve().relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        value = path.resolve()
    return str(value).replace("\\", "/")


def prepare_watermark_input(
    pdf_path: Path,
    out_dir: Path,
    rendered_pages: Sequence[dict],
) -> tuple[Path, dict[int, WatermarkDetection], Path]:
    """Detect watermarks and return a prediction PDF only when pages were cleaned."""
    out_dir.mkdir(parents=True, exist_ok=True)
    page_images = [
        (int(page["page"]), _resolve_page_image(page["page_image"]))
        for page in rendered_pages
    ]
    detections = detect_orange_gjb_watermarks(page_images)
    by_page = {item.page: item for item in detections}
    clean_dir = out_dir / "pages_clean"
    for stale in clean_dir.glob("p*.png") if clean_dir.exists() else ():
        stale.unlink(missing_ok=True)

    for page, image_path in page_images:
        detection = by_page.get(page)
        if detection is None:
            continue
        cleaned_path = clean_dir / f"p{page:03d}.png"
        clean_orange_watermark(image_path, cleaned_path)
        detection.cleaned_image = _stored_path(cleaned_path)

    metadata_path = out_dir / "watermarks.json"
    metadata_path.write_text(
        json.dumps(
            {
                "profile": PROFILE_NAME,
                "detected_pages": [asdict(item) for item in detections],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if not detections:
        return pdf_path, by_page, metadata_path

    prediction_pdf = out_dir / "_watermark_cleaned_input.pdf"
    prediction_pdf.unlink(missing_ok=True)
    source = fitz.open(str(pdf_path))
    target = fitz.open()
    try:
        for index, source_page in enumerate(source):
            page_num = index + 1
            detection = by_page.get(page_num)
            if detection is None:
                target.insert_pdf(source, from_page=index, to_page=index)
                continue
            clean_image = _resolve_page_image(detection.cleaned_image or "")
            page = target.new_page(
                width=source_page.rect.width,
                height=source_page.rect.height,
            )
            page.insert_image(page.rect, filename=str(clean_image), keep_proportion=False)
        target.save(prediction_pdf, garbage=4, deflate=True)
    finally:
        target.close()
        source.close()
    return prediction_pdf, by_page, metadata_path
