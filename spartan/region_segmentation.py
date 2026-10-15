"""Stage 3 - Region segmentation (the core of SPARTAN).

Takes a page (or column) image and splits it into ``pre_text``, ``table_N``,
``inter_text_N`` and ``post_text`` segments:

1. **Crop outer boundaries** - a 4-sided contour covering >= 75% of the image is
   treated as a frame/scan border and cut away with a 25 px safety margin.
2. **Boundary-based detector** - tables with visible boundaries: morphological
   line extraction (25x1 / 1x25 kernels), run with and without an extra 3x3
   dilation (the path yielding more cells wins), 4-sided convex cell contours
   (w > 100, h > 20), clustering of cell rows whose gap is < 20 px, and
   acceptance of regions with w >= 500 and h >= 100 (paper Table 3).
3. **Text-based detector** - borderless tables formed purely by text
   arrangement: phrases stacked over each other with left-edge alignment
   (5 px tolerance), falling back to centre alignment (and, as an extension, right
   alignment for numeric columns). At least two such columns whose tops align
   form a table region. **Pseudo boundaries** are then drawn around the text so
   the borderless table is handled by exactly the same downstream logic as a
   bordered one.
4. **Table unification** - a text-based detection overlapping a boundary-based
   one is discarded (both point at the same table).
5. **Segmentation** into pre/inter/post text and table segments.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Sequence

import cv2
import numpy as np

from .config import PipelineConfig
from .models import BBox, Segment, SegmentKind, TableCandidate, Word, group_words_into_lines, words_in_box
from .pre_processing import preprocess_image

log = logging.getLogger(__name__)

_LIST_MARKER = re.compile(r"^(\(?[0-9]{1,2}[.)]|\(?[a-zA-Z][.)]|\(?[ivxIVX]{1,4}[.)]|[-–•*·o▪■●])(\s+\(?[0-9a-zA-Z]{1,3}[.)])*$")


# --------------------------------------------------------------------------- #
# 1. Cropping outer boundaries
# --------------------------------------------------------------------------- #
def crop_outer_boundary(image: np.ndarray, binary: np.ndarray, config: PipelineConfig) -> BBox:
    """Return the crop box (image coordinates) that removes a page frame, if any."""
    h, w = image.shape[:2]
    p = config.region_params
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best: tuple[int, int, int, int] | None = None
    best_area = 0
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            x, y, cw, ch = cv2.boundingRect(approx)
            if cw * ch > best_area:
                best_area, best = cw * ch, (x, y, cw, ch)
    if best is not None and best_area >= p.crop_min_area_ratio * w * h:
        x, y, cw, ch = best
        m = config.px(p.crop_margin)
        box = BBox(x + m, y + m, x + cw - m, y + ch - m).clip(w, h)
        if box.width > 0 and box.height > 0:
            log.debug("Outer frame detected; cropping to %s", box.to_list())
            return box
    return BBox(0, 0, w, h)


# --------------------------------------------------------------------------- #
# 2. Boundary-based detector
# --------------------------------------------------------------------------- #
def _detect_cells(binary: np.ndarray, config: PipelineConfig, use_dilation: bool) -> list[BBox]:
    p = config.region_params
    k = config.px(p.line_kernel)
    horiz = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))
    vert = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, k)))
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    if use_dilation:
        horiz = cv2.dilate(horiz, kernel3, iterations=1)
        vert = cv2.dilate(vert, kernel3, iterations=1)
    grid = cv2.add(horiz, vert)
    if use_dilation:
        grid = cv2.dilate(grid, kernel3, iterations=1)

    min_w, min_h = config.px(p.min_cell_width), config.px(p.min_cell_height)
    contours, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cells = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            x, y, w, h = cv2.boundingRect(approx)
            if w > min_w and h > min_h:
                cells.append(BBox.from_xywh(x, y, w, h))
    return cells


def find_boundary_tables(binary: np.ndarray, config: PipelineConfig) -> tuple[list[BBox], list[BBox]]:
    """Detect tables with visible boundaries. Returns ``(tables, cells)``."""
    p = config.region_params
    without = _detect_cells(binary, config, use_dilation=False)
    with_dil = _detect_cells(binary, config, use_dilation=True)
    cells = with_dil if len(with_dil) > len(without) else without
    if not cells:
        return [], []

    cells.sort(key=lambda c: (c.y0, c.x0))
    gap = config.px(p.cell_gap_threshold)
    clusters: list[list[BBox]] = [[cells[0]]]
    bottom = cells[0].y1
    for cell in cells[1:]:
        if cell.y0 > bottom + gap:
            clusters.append([cell])
            bottom = cell.y1
        else:
            clusters[-1].append(cell)
            bottom = max(bottom, cell.y1)

    min_w, min_h = config.px(p.min_table_width), config.px(p.min_table_height)
    tables = [box for box in (BBox.union_all(c) for c in clusters) if box.width >= min_w and box.height >= min_h]
    return tables, cells


# --------------------------------------------------------------------------- #
# 3. Text-based (borderless) detector
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class _Phrase:
    line: int
    bbox: BBox
    words: list[Word] = field(default_factory=list)


def _line_phrases(lines: list[list[Word]], config: PipelineConfig) -> list[list[_Phrase]]:
    """Split each text line into phrases separated by column-sized whitespace."""
    p = config.region_params
    result = []
    for li, line in enumerate(lines):
        height = float(np.median([w.bbox.height for w in line]))
        threshold = max(config.px(p.min_column_gap), p.phrase_gap_ratio * height)
        phrases: list[_Phrase] = []
        for word in line:
            if phrases and word.bbox.x0 - phrases[-1].bbox.x1 <= threshold:
                phrases[-1].bbox = phrases[-1].bbox.union(word.bbox)
                phrases[-1].words.append(word)
            else:
                phrases.append(_Phrase(li, word.bbox, [word]))
        result.append(phrases)
    return result


def _aligned(a: BBox, b: BBox, tol: float, use_right: bool) -> bool:
    if abs(a.x0 - b.x0) <= tol:          # paper: left-edge alignment
        return True
    if abs(a.cx - b.cx) <= tol:          # paper: fallback to centre alignment
        return True
    return use_right and abs(a.x1 - b.x1) <= tol


def find_text_tables(words: Sequence[Word], width: int, height: int, config: PipelineConfig) -> list[TableCandidate]:
    """Detect borderless tables from word positions and build their pseudo grid."""
    p = config.region_params
    lines = group_words_into_lines(words)
    if len(lines) < p.min_rows:
        return []
    phrases = _line_phrases(lines, config)
    line_h = [float(np.median([w.bbox.height for w in ln])) for ln in lines]
    multi = [i for i, ph in enumerate(phrases) if len(ph) >= 2]
    tol = config.px(p.align_tolerance)

    # Link phrases of consecutive multi-phrase lines that are vertically stacked.
    parent: dict[int, int] = {}
    nodes: list[_Phrase] = []
    index: dict[int, int] = {}

    def node(ph: _Phrase) -> int:
        key = id(ph)
        if key not in index:
            index[key] = len(nodes)
            nodes.append(ph)
            parent[index[key]] = index[key]
        return index[key]

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a_idx, b_idx in zip(multi[:-1], multi[1:]):
        top_a = min(w.bbox.y0 for w in lines[a_idx])
        top_b = min(w.bbox.y0 for w in lines[b_idx])
        if top_b - top_a > 3.5 * max(line_h[a_idx], line_h[b_idx]):
            continue  # too far apart: not consecutive table rows
        used = set()
        for pa in phrases[a_idx]:
            match = next(
                (pb for pb in phrases[b_idx] if id(pb) not in used and _aligned(pa.bbox, pb.bbox, tol, p.use_right_alignment)),
                None,
            )
            if match is not None:
                used.add(id(match))
                parent[find(node(match))] = find(node(pa))

    chains: dict[int, list[_Phrase]] = {}
    for i in range(len(nodes)):
        chains.setdefault(find(i), []).append(nodes[i])
    columns = [sorted(c, key=lambda ph: ph.line) for c in chains.values() if len(c) >= 2]
    if len(columns) < p.min_columns:
        return []

    # Group columns whose line ranges overlap substantially (paper: tops share the same y).
    columns.sort(key=lambda c: (c[0].line, c[0].bbox.x0))
    groups: list[list[list[_Phrase]]] = []
    for col in columns:
        start, end = col[0].line, col[-1].line
        for group in groups:
            g_start = min(c[0].line for c in group)
            g_end = max(c[-1].line for c in group)
            overlap = min(end, g_end) - max(start, g_start) + 1
            if overlap >= min(p.min_rows, end - start + 1) and abs(start - g_start) <= 1:
                group.append(col)
                break
        else:
            groups.append([col])

    candidates = []
    min_w = config.px(p.min_text_table_width)
    for group in groups:
        if len(group) < p.min_columns:
            continue
        first = min(c[0].line for c in group)
        last = max(c[-1].line for c in group)
        if last - first + 1 < p.min_rows:
            continue
        box = BBox.union_all(ph.bbox for col in group for ph in col)
        # Include every phrase of the covered lines that horizontally overlaps the table.
        for li in range(first, last + 1):
            for ph in phrases[li]:
                if ph.bbox.x1 > box.x0 and ph.bbox.x0 < box.x1:
                    box = box.union(ph.bbox)
        if box.width < min_w:
            continue
        cand = _build_pseudo_grid(box, words, width, height, config)
        if cand is not None:
            candidates.append(cand)
    return _remove_nested(candidates)


def _remove_nested(cands: list[TableCandidate]) -> list[TableCandidate]:
    kept: list[TableCandidate] = []
    for c in sorted(cands, key=lambda t: -t.bbox.area):
        if not any(k.bbox.overlap_ratio(c.bbox) > 0.5 for k in kept):
            kept.append(c)
    return kept


def _build_pseudo_grid(
    box: BBox, words: Sequence[Word], width: int, height: int, config: PipelineConfig
) -> TableCandidate | None:
    """Compute column and row separators for a borderless table region."""
    p = config.region_params
    table_words = [w for w in words if box.contains_point(w.bbox.cx, w.bbox.cy)]
    lines = group_words_into_lines(table_words)
    if len(lines) < p.min_rows:
        return None
    phrases = _line_phrases(lines, config)

    # Column separators: x ranges not covered by any phrase (allowing one spanning line).
    x0, x1 = int(box.x0), int(np.ceil(box.x1))
    occupancy = np.zeros(x1 - x0 + 1, dtype=np.int32)
    for line in phrases:
        for ph in line:
            occupancy[int(ph.bbox.x0) - x0:int(np.ceil(ph.bbox.x1)) - x0] += 1
    min_sep = config.px(10)
    separators: list[float] = []
    for allowed in (0, 1):
        if allowed and len(lines) < 4:
            break
        free = occupancy <= allowed
        separators = []
        start = None
        for i, f in enumerate(free):
            if f and start is None:
                start = i
            elif not f and start is not None:
                if start > 0 and i - start >= min_sep:
                    separators.append(x0 + (start + i) / 2.0)
                start = None
        if len(separators) + 1 >= p.min_columns:
            break
    if len(separators) + 1 < p.min_columns:
        return None

    # Row separators: a line starts a new row when it has text in the first column or
    # in at least half of the columns; other lines continue a wrapped row.
    edges = [box.x0, *separators, box.x1]
    n_cols = len(edges) - 1

    def columns_of(line: list[_Phrase]) -> set[int]:
        cols = set()
        for ph in line:
            for c in range(n_cols):
                if ph.bbox.cx >= edges[c] and ph.bbox.cx < edges[c + 1]:
                    cols.add(c)
        return cols

    # A "first column" made only of list markers (1. 2. / a) b) / bullets) is a list, not a table.
    first_col = [
        " ".join(w.text for w in ph.words) for line in phrases for ph in line if ph.bbox.cx < edges[1]
    ]
    if first_col and sum(bool(_LIST_MARKER.match(t)) for t in first_col) >= 0.8 * len(first_col):
        return None

    rows: list[list[int]] = []
    for li, line in enumerate(phrases):
        cols = columns_of(line)
        if not rows or 0 in cols or len(cols) >= max(2, (n_cols + 1) // 2):
            rows.append([li])
        else:
            rows[-1].append(li)
    row_seps = []
    for prev, nxt in zip(rows[:-1], rows[1:]):
        bottom = max(w.bbox.y1 for li in prev for w in lines[li])
        top = min(w.bbox.y0 for li in nxt for w in lines[li])
        row_seps.append((bottom + top) / 2.0)

    pad = config.px(p.pseudo_padding)
    padded = box.expand(pad).clip(width, height)
    return TableCandidate(bbox=padded, method="text", pseudo_columns=separators, pseudo_rows=row_seps)


def draw_pseudo_boundaries(table_image: np.ndarray, cand: TableCandidate, origin: BBox, config: PipelineConfig) -> np.ndarray:
    """Draw synthetic ruling lines on a borderless table crop (paper Fig. 9)."""
    img = table_image.copy()
    h, w = img.shape[:2]
    t = config.px(config.region_params.pseudo_line_thickness)
    color = (0, 0, 0)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), color, t)
    for x in cand.pseudo_columns:
        xi = int(round(x - origin.x0))
        cv2.line(img, (xi, 0), (xi, h - 1), color, t)
    for y in cand.pseudo_rows:
        yi = int(round(y - origin.y0))
        cv2.line(img, (0, yi), (w - 1, yi), color, t)
    return img


# --------------------------------------------------------------------------- #
# 4. Unification and 5. segmentation
# --------------------------------------------------------------------------- #
def unify_tables(boundary: list[BBox], text: list[TableCandidate]) -> list[TableCandidate]:
    """Boundary tables win; text tables overlapping them are discarded."""
    unified = [TableCandidate(bbox=b, method="boundary") for b in boundary]
    for cand in text:
        if any(cand.bbox.intersects(b) for b in boundary):
            continue
        unified.append(cand)
    return unified


def detect_tables(image: np.ndarray, words: Sequence[Word], config: PipelineConfig) -> tuple[BBox, list[TableCandidate], dict]:
    """Run cropping + both detectors + unification. Returns ``(crop_box, tables, debug)``."""
    _, binary = preprocess_image(image, config)
    crop = crop_outer_boundary(image, binary, config).to_int()
    rows, cols = crop.as_slice()
    cropped_binary = binary[rows, cols]
    cropped_words = words_in_box(words, crop)

    boundary, cells = find_boundary_tables(cropped_binary, config)
    text = find_text_tables(cropped_words, int(crop.width), int(crop.height), config)
    tables = unify_tables(boundary, text)
    debug = {"boundary": boundary, "cells": cells, "text": [t.bbox for t in text]}
    return crop, tables, debug


def segment_region(
    image: np.ndarray, words: Sequence[Word], config: PipelineConfig
) -> tuple[list[Segment], BBox, dict]:
    """Split an image into ordered pre-text / table / inter-text / post-text segments.

    Returns ``(segments, crop_box, debug)``; segment boxes are relative to the
    input image (the crop offset is already applied).
    """
    crop, tables, debug = detect_tables(image, words, config)
    tables.sort(key=lambda t: (t.bbox.y0, t.bbox.x0))
    W = int(crop.width)

    segments: list[Segment] = []

    def add_text(seg_id: str, kind: SegmentKind, y0: float, y1: float) -> None:
        if y1 - y0 < 2:
            return
        box = BBox(0, y0, W, y1).translate(crop.x0, crop.y0).to_int()
        seg_words = words_in_box(words, box)
        if not seg_words:
            return
        rows, cols = box.as_slice()
        segments.append(Segment(seg_id, kind, box, image[rows, cols], seg_words))

    cursor = 0.0
    inter_count = 0
    for i, table in enumerate(tables):
        if i == 0:
            add_text("pre_text", SegmentKind.PRE_TEXT, 0, table.bbox.y0)
        elif table.bbox.y0 > cursor:
            inter_count += 1
            add_text(f"inter_text_{inter_count}", SegmentKind.INTER_TEXT, cursor, table.bbox.y0)

        box = table.bbox.translate(crop.x0, crop.y0).clip(image.shape[1], image.shape[0]).to_int()
        rows, cols = box.as_slice()
        table_img = image[rows, cols]
        pseudo = table.method == "text"
        if pseudo:
            table_img = draw_pseudo_boundaries(table_img, table, table.bbox, config)
        segments.append(Segment(
            id=f"table_{i + 1}",
            kind=SegmentKind.TABLE,
            bbox=box,
            image=table_img,
            words=words_in_box(words, box),
            detection_method=table.method,
            pseudo_bordered=pseudo,
        ))
        cursor = max(cursor, table.bbox.y1)

    if tables:
        add_text("post_text", SegmentKind.POST_TEXT, cursor, crop.height)
    else:
        add_text("pre_text", SegmentKind.PRE_TEXT, 0, crop.height)

    debug["tables"] = tables
    return segments, crop, debug
