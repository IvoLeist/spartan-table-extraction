"""Debug visualisations written with ``--save`` (OpenCV only, no GUI required).

Colours follow the paper's figures: column-merged cells pink, row-merged cells
yellow, cells merged in both directions blue; header rows are tinted green.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .models import BBox, MergeType, ParsedTable, Table, TableCandidate

_MERGE_COLORS = {
    MergeType.COLUMN: (203, 192, 255),   # pink (BGR)
    MergeType.ROW: (0, 255, 255),        # yellow
    MergeType.BOTH: (255, 128, 0),       # blue
}


def _rect(img: np.ndarray, box: BBox, color, thickness: int = 3) -> None:
    b = box.to_int()
    cv2.rectangle(img, (int(b.x0), int(b.y0)), (int(b.x1), int(b.y1)), color, thickness)


def _fill(img: np.ndarray, box: BBox, color, alpha: float = 0.35) -> None:
    b = box.clip(img.shape[1], img.shape[0]).to_int()
    x0, y0, x1, y1 = int(b.x0), int(b.y0), int(b.x1), int(b.y1)
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    overlay = np.full_like(roi, color)
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)


def draw_columns(image: np.ndarray, gaps, regions: Sequence[BBox]) -> np.ndarray:
    out = image.copy()
    for g in gaps:
        _fill(out, BBox(g.x0, g.y0, g.x1, g.y1), (0, 0, 255), 0.5)
    for r in regions:
        _rect(out, r, (255, 0, 0), 4)
    return out


def draw_regions(image: np.ndarray, crop: BBox, debug: dict) -> np.ndarray:
    out = image.copy()
    _rect(out, crop, (128, 128, 128), 2)
    for cell in debug.get("cells", []):
        _fill(out, cell.translate(crop.x0, crop.y0), (0, 200, 0), 0.15)
    for table in debug.get("tables", []):
        assert isinstance(table, TableCandidate)
        color = (0, 0, 255) if table.method == "boundary" else (255, 0, 255)
        _rect(out, table.bbox.translate(crop.x0, crop.y0), color, 6)
    return out


def draw_parsed_table(image: np.ndarray, parsed: ParsedTable, tables: Sequence[Table]) -> np.ndarray:
    out = image.copy()
    header_depth = tables[0].header_rows if tables else 0
    for cell in parsed.cells:
        if cell.row < header_depth:
            _fill(out, cell.bbox, (0, 200, 0), 0.25)
        color = _MERGE_COLORS.get(cell.merge_type)
        if color is not None:
            _fill(out, cell.bbox, color)
    for x1, y1, x2, y2 in parsed.horizontal_lines + parsed.vertical_lines:
        cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    for cell in parsed.cells:
        _rect(out, cell.bbox, (255, 0, 0), 1)
    return out


def save(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)
