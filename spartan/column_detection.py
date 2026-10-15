"""Stage 2 (optional) - Column detection.

Splits multi-column page layouts (e.g. scientific papers) into column segments
by analysing continuous vertical whitespace, following the paper:

1. Find vertical whitespace gaps (fully vectorised with morphology: a gap is the
   union of white rectangles at least ``min_gap_width`` wide and
   ``min_gap_height`` tall). Gaps shorter than 200 px (and shorter than a
   fraction of the content height) are ignored.
2. A gap is valid only if there is text at most 15 px away on *both* sides,
   which rejects page margins.
3. If a longer gap exists in the proximity of a gap, the shorter one is ignored.
4. Gaps whose x positions are closer than 15 px are merged.
5. Gaps overlapping a detected table are discarded: the whitespace between the
   columns of a borderless table is not a layout gutter (this is the false
   positive the paper reports for single-column documents).
6. The page is cut into horizontal bands; inside each band the active gaps split
   it into columns. Segments are returned in reading order.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from .config import PipelineConfig
from .models import BBox, ImageRegion, PageImage, Word, words_in_box

log = logging.getLogger(__name__)


@dataclass(slots=True)
class VerticalGap:
    x: float          # split position
    x0: float
    x1: float
    y0: float
    y1: float

    @property
    def height(self) -> float:
        return self.y1 - self.y0


def _content_bbox(ink: np.ndarray) -> BBox | None:
    ys, xs = np.nonzero(ink)
    if xs.size == 0:
        return None
    return BBox(float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def find_vertical_gaps(
    ink: np.ndarray, config: PipelineConfig, exclusion_boxes: Sequence[BBox] = ()
) -> list[VerticalGap]:
    """Detect valid column gutters in an ink mask (255 = ink)."""
    p = config.column_params
    content = _content_bbox(ink)
    if content is None:
        return []
    c = content.to_int()
    cx0, cy0, cx1, cy1 = int(c.x0), int(c.y0), int(c.x1), int(c.y1)
    region = ink[cy0:cy1, cx0:cx1]
    content_h = cy1 - cy0

    min_h = max(config.px(p.min_gap_height), int(p.min_gap_height_ratio * content_h))
    min_w = config.px(p.min_gap_width)
    if content_h < min_h or region.shape[1] < 3 * min_w:
        return []

    white = (region == 0).astype(np.uint8)
    gaps_mask = cv2.morphologyEx(white, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (min_w, min_h)))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(gaps_mask, connectivity=4)
    prox = config.px(p.text_proximity)
    candidates: list[VerticalGap] = []
    for label in range(1, n):
        x, y, w, h = stats[label, :4]
        # Components touching the left/right content edge are margins or ragged line ends.
        if x <= 0 or x + w >= region.shape[1]:
            continue
        comp = labels[y:y + h, x:x + w] == label
        col_heights = comp.sum(axis=0)
        best = np.flatnonzero(col_heights == col_heights.max())
        split_local = x + (best[0] + best[-1]) / 2.0
        rows = np.flatnonzero(comp[:, int(round(split_local - x))])
        gy0, gy1 = y + rows.min(), y + rows.max() + 1

        # Paper: there must be text within `text_proximity` px on both sides of the gap.
        gap_cols = np.flatnonzero(comp[rows.min():rows.max() + 1].any(axis=0)) + x
        left_edge, right_edge = gap_cols.min(), gap_cols.max() + 1
        left_strip = region[gy0:gy1, max(0, left_edge - prox):left_edge]
        right_strip = region[gy0:gy1, right_edge:right_edge + prox]
        need = p.min_side_ink_ratio * (gy1 - gy0)
        if left_strip.size == 0 or right_strip.size == 0:
            continue
        if left_strip.any(axis=1).sum() < need or right_strip.any(axis=1).sum() < need:
            continue

        candidates.append(VerticalGap(
            x=cx0 + split_local, x0=cx0 + left_edge, x1=cx0 + right_edge, y0=cy0 + gy0, y1=cy0 + gy1,
        ))

    # Paper: drop a gap if a longer one is nearby.
    supp = config.px(p.suppression_distance)
    kept = []
    for g in candidates:
        overlaps_longer = any(
            o is not g and o.height > g.height and abs(o.x - g.x) <= supp and o.y0 < g.y1 and g.y0 < o.y1
            for o in candidates
        )
        if not overlaps_longer:
            kept.append(g)

    # Paper: merge gaps sharing (almost) the same x.
    kept.sort(key=lambda g: g.x)
    merged: list[VerticalGap] = []
    merge_d = config.px(p.merge_distance)
    for g in kept:
        if merged and abs(g.x - merged[-1].x) <= merge_d:
            m = merged[-1]
            merged[-1] = VerticalGap((m.x + g.x) / 2, min(m.x0, g.x0), max(m.x1, g.x1), min(m.y0, g.y0), max(m.y1, g.y1))
        else:
            merged.append(g)

    # Whitespace inside tables is not a layout gutter.
    result = []
    for g in merged:
        gbox = BBox(g.x0, g.y0, g.x1, g.y1)
        if any(gbox.intersects(t) for t in exclusion_boxes):
            log.debug("Column gap at x=%.0f discarded: overlaps a table", g.x)
            continue
        result.append(g)
    return result


def segment_by_gaps(width: int, height: int, gaps: Sequence[VerticalGap]) -> list[BBox]:
    """Cut the page into column boxes: horizontal bands, split by the gaps active in each band."""
    if not gaps:
        return [BBox(0, 0, width, height)]
    cuts = sorted({0, height, *[int(g.y0) for g in gaps], *[int(g.y1) for g in gaps]})
    boxes: list[BBox] = []
    for top, bottom in zip(cuts[:-1], cuts[1:]):
        if bottom - top <= 0:
            continue
        mid = (top + bottom) / 2
        xs = sorted(g.x for g in gaps if g.y0 <= mid < g.y1)
        edges = [0.0, *xs, float(width)]
        for left, right in zip(edges[:-1], edges[1:]):
            if right - left > 1:
                boxes.append(BBox(left, top, right, bottom))
    # Merge vertically adjacent boxes with identical x-extent (same column across bands).
    merged: list[BBox] = []
    for box in boxes:
        target = next(
            (i for i, m in enumerate(merged) if abs(m.x0 - box.x0) < 1 and abs(m.x1 - box.x1) < 1 and abs(m.y1 - box.y0) < 1),
            None,
        )
        if target is None:
            merged.append(box)
        else:
            merged[target] = merged[target].union(box)
    # Reading order: by band top, then left to right.
    merged.sort(key=lambda b: (b.y0, b.x0))
    return merged


def detect_columns(
    page: PageImage,
    words: Sequence[Word],
    config: PipelineConfig,
    table_boxes: Sequence[BBox] = (),
) -> tuple[list[ImageRegion], list[VerticalGap]]:
    """Split a page into column regions. Returns ``(regions, gaps)``."""
    ink = (page.gray < config.preprocess_params.ink_threshold).astype(np.uint8) * 255
    gaps = find_vertical_gaps(ink, config, table_boxes)
    boxes = segment_by_gaps(page.width, page.height, gaps)

    margin = config.px(config.column_params.segment_margin)
    regions: list[ImageRegion] = []
    for box in boxes:
        padded = box.expand(margin).clip(page.width, page.height).to_int()
        rows, cols = padded.as_slice()
        if not ink[rows, cols].any():
            continue  # blank segment
        regions.append(ImageRegion(
            index=len(regions),
            bbox=padded,
            image=page.bgr[rows, cols],
            words=words_in_box(words, padded),
        ))
    log.info("Page %d: %d column gap(s), %d column segment(s)", page.page_number, len(gaps), len(regions))
    return regions, gaps
