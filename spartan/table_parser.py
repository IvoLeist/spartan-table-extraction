"""Stage 4 - Table parsing.

Recovers a reliable cell grid from a table segment image:

1. **Text masking** - the words of the table (known from the text layer or OCR)
   are painted over with the local background colour, so glyph strokes (``l``,
   ``I``, ``1`` ...) cannot be mistaken for ruling lines. Using the local
   background (not white) keeps fill-delimited tables intact.
2. **LSD line detection** - OpenCV's Line Segment Detector works directly on
   gray-level intensities and recovers sub-pixel accurate segments, robust to
   mild skew and broken borders. Near-horizontal and near-vertical segments are
   kept.
3. **Merging** - segments sharing the same y (horizontal) or x (vertical) are
   merged into single lines and collinear pieces with small gaps are joined.
   Short segments that never took part in a merge are discarded (underlines,
   noise), as are horizontal lines that touch no vertical line at either end.
4. **Grid construction** - every line position defines a grid boundary
   (the paper's "extended lines"). For each pair of neighbouring atomic cells we
   check whether a real line covers at least 50% of their shared edge; where it
   does not, the atomic cells are merged. Each connected group becomes one cell,
   so a cell crossed by *k* extended lines spans *k + 1* rows/columns, exactly
   as described in the paper.
5. **Refinement** - empty outer rows/columns that are not supported by real
   lines are trimmed, giving a refined table region.
"""

from __future__ import annotations

import logging
from typing import Sequence

import cv2
import numpy as np

from .config import PipelineConfig
from .models import BBox, GridCell, ParsedTable, Word

log = logging.getLogger(__name__)

Line = tuple[float, float, float, float]   # x1, y1, x2, y2


# --------------------------------------------------------------------------- #
# 1. Text masking
# --------------------------------------------------------------------------- #
def mask_text(gray: np.ndarray, words: Sequence[Word], pad: int) -> np.ndarray:
    """Paint word boxes with the median colour of a thin ring around them."""
    out = gray.copy()
    h, w = gray.shape
    for word in words:
        b = word.ink.expand(1).clip(w, h).to_int()
        x0, y0, x1, y1 = int(b.x0), int(b.y0), int(b.x1), int(b.y1)
        if x1 <= x0 or y1 <= y0:
            continue
        rx0, ry0, rx1, ry1 = max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)
        ring = np.concatenate([
            gray[ry0:y0, rx0:rx1].ravel(), gray[y1:ry1, rx0:rx1].ravel(),
            gray[y0:y1, rx0:x0].ravel(), gray[y0:y1, x1:rx1].ravel(),
        ])
        out[y0:y1, x0:x1] = int(np.median(ring)) if ring.size else 255
    return out


# --------------------------------------------------------------------------- #
# 2. LSD detection
# --------------------------------------------------------------------------- #
def detect_segments(gray: np.ndarray, config: PipelineConfig) -> tuple[list[Line], list[Line]]:
    """Return ``(horizontal, vertical)`` line segments found by LSD."""
    p = config.parser_params
    lsd = cv2.createLineSegmentDetector()
    lines = lsd.detect(gray)[0]
    horizontal, vertical = [], []
    if lines is None:
        return horizontal, vertical
    min_v, min_h = config.px(p.min_vertical_segment), config.px(p.min_horizontal_segment)
    skew = max(p.max_skew, 2.0 * config.scale)
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):   # (N,1,4) in OpenCV 4, (N,4) in 5
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        if dy <= skew and dx >= min_h:
            horizontal.append((min(x1, x2), (y1 + y2) / 2, max(x1, x2), (y1 + y2) / 2))
        elif dx <= skew and dy >= min_v:
            vertical.append(((x1 + x2) / 2, min(y1, y2), (x1 + x2) / 2, max(y1, y2)))
    return horizontal, vertical


# --------------------------------------------------------------------------- #
# 3. Merging
# --------------------------------------------------------------------------- #
def _merge_collinear(
    segments: Sequence[Line], horizontal: bool, cluster_tol: float, gap_tol: float
) -> list[tuple[float, list[tuple[float, float]], int]]:
    """Cluster segments by their constant coordinate and join their extents.

    Returns ``[(position, [(start, end), ...], n_segments), ...]``.
    """
    if not segments:
        return []
    key = (lambda s: s[1]) if horizontal else (lambda s: s[0])
    span = (lambda s: (s[0], s[2])) if horizontal else (lambda s: (s[1], s[3]))
    ordered = sorted(segments, key=key)
    clusters: list[list[Line]] = [[ordered[0]]]
    for seg in ordered[1:]:
        if key(seg) - key(clusters[-1][-1]) <= cluster_tol:
            clusters[-1].append(seg)
        else:
            clusters.append([seg])

    merged = []
    for cluster in clusters:
        position = float(np.average([key(s) for s in cluster], weights=[span(s)[1] - span(s)[0] + 1 for s in cluster]))
        intervals = sorted(span(s) for s in cluster)
        joined: list[list[float]] = [list(intervals[0])]
        for start, end in intervals[1:]:
            if start - joined[-1][1] <= gap_tol:
                joined[-1][1] = max(joined[-1][1], end)
            else:
                joined.append([start, end])
        merged.append((position, [(a, b) for a, b in joined], len(cluster)))
    return merged


def build_lines(
    horizontal: Sequence[Line], vertical: Sequence[Line], width: int, height: int, config: PipelineConfig
) -> tuple[list[Line], list[Line]]:
    """Merge raw segments into clean lines; add the table border; drop isolated lines."""
    p = config.parser_params
    gap = config.px(p.gap_tolerance)
    h_merged = _merge_collinear(horizontal, True, config.px(p.horizontal_cluster_tolerance), gap)
    v_merged = _merge_collinear(vertical, False, config.px(p.vertical_cluster_tolerance), gap)

    min_h, min_v = config.px(p.min_horizontal_line), config.px(p.min_vertical_line)
    h_lines: list[Line] = []
    for y, intervals, count in h_merged:
        for a, b in intervals:
            # Short lines that were never merged with anything are treated as noise.
            if b - a >= min_h or (count > 1 and b - a >= min_h / 2):
                h_lines.append((a, y, b, y))
    v_lines: list[Line] = []
    for x, intervals, count in v_merged:
        for a, b in intervals:
            if b - a >= min_v or (count > 1 and b - a >= min_v / 2):
                v_lines.append((x, a, x, b))

    # The segment is cropped to the table, so its border acts as the outer frame.
    w1, h1 = float(width - 1), float(height - 1)
    v_lines += [(0.0, 0.0, 0.0, h1), (w1, 0.0, w1, h1)]
    h_lines += [(0.0, 0.0, w1, 0.0), (0.0, h1, w1, h1)]

    # Drop horizontal lines that touch no vertical line at either end
    # (underlines, rules inside text), unless they span almost the whole table.
    tol = config.px(p.endpoint_tolerance)

    def touches(x: float, y: float) -> bool:
        return any(abs(vx - x) <= tol and vy0 - tol <= y <= vy1 + tol for vx, vy0, _, vy1 in v_lines)

    kept_h = [
        ln for ln in h_lines
        if (ln[2] - ln[0]) >= 0.9 * width or (touches(ln[0], ln[1]) and touches(ln[2], ln[1]))
    ]
    return kept_h, v_lines


# --------------------------------------------------------------------------- #
# 4. Grid construction
# --------------------------------------------------------------------------- #
def _boundaries(positions: Sequence[float], extent: float, min_gap: float) -> list[float]:
    """Sorted, de-duplicated boundary positions including both borders."""
    pts = sorted({0.0, float(extent - 1), *positions})
    result: list[float] = []
    for p in pts:
        if not result or p - result[-1] >= min_gap:
            result.append(p)
        else:
            result[-1] = (result[-1] + p) / 2 if 0 < result[-1] < extent - 1 else result[-1]
    result[0], result[-1] = 0.0, float(extent - 1)
    return result


def _coverage(lines: Sequence[tuple[float, float, float]], pos: float, a: float, b: float, tol: float) -> float:
    """Fraction of interval [a, b] at position ``pos`` covered by ``lines`` (pos, start, end)."""
    if b <= a:
        return 0.0
    pieces = sorted((max(a, s), min(b, e)) for p, s, e in lines if abs(p - pos) <= tol and e > a and s < b)
    covered, cursor = 0.0, a
    for s, e in pieces:
        if e <= cursor:
            continue
        covered += e - max(s, cursor)
        cursor = e
    return covered / (b - a)


def build_grid(h_lines: Sequence[Line], v_lines: Sequence[Line], width: int, height: int, config: PipelineConfig) -> ParsedTable:
    p = config.parser_params
    min_cell = config.px(p.min_cell_size)
    xs = _boundaries([ln[0] for ln in v_lines], width, min_cell)
    ys = _boundaries([ln[1] for ln in h_lines], height, min_cell)
    n_rows, n_cols = len(ys) - 1, len(xs) - 1
    tol = config.px(p.vertical_cluster_tolerance)
    v_idx = [(ln[0], ln[1], ln[3]) for ln in v_lines]
    h_idx = [(ln[1], ln[0], ln[2]) for ln in h_lines]
    ratio = p.edge_coverage_ratio

    parent = list(range(n_rows * n_cols))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for r in range(n_rows):
        for c in range(n_cols):
            here = r * n_cols + c
            if c + 1 < n_cols and _coverage(v_idx, xs[c + 1], ys[r], ys[r + 1], tol) < ratio:
                union(here, here + 1)
            if r + 1 < n_rows and _coverage(h_idx, ys[r + 1], xs[c], xs[c + 1], tol) < ratio:
                union(here, here + n_cols)

    # Make every group rectangular (broken borders can create L-shapes).
    changed = True
    while changed:
        changed = False
        groups: dict[int, list[int]] = {}
        for i in range(n_rows * n_cols):
            groups.setdefault(find(i), []).append(i)
        for members in groups.values():
            rs = [m // n_cols for m in members]
            cs = [m % n_cols for m in members]
            r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
            if len(members) != (r1 - r0 + 1) * (c1 - c0 + 1):
                for r in range(r0, r1 + 1):
                    for c in range(c0, c1 + 1):
                        if find(r * n_cols + c) != find(members[0]):
                            union(members[0], r * n_cols + c)
                            changed = True

    groups = {}
    for i in range(n_rows * n_cols):
        groups.setdefault(find(i), []).append(i)
    cells = []
    for members in sorted(groups.values(), key=lambda m: (min(m) // n_cols, min(m) % n_cols)):
        rs = [m // n_cols for m in members]
        cs = [m % n_cols for m in members]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        cells.append(GridCell(
            id=len(cells), row=r0, col=c0, row_span=r1 - r0 + 1, col_span=c1 - c0 + 1,
            bbox=BBox(xs[c0], ys[r0], xs[c1 + 1], ys[r1 + 1]),
        ))
    return ParsedTable(width=width, height=height, xs=xs, ys=ys, cells=cells,
                       horizontal_lines=list(h_lines), vertical_lines=list(v_lines))


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def parse_table(image: np.ndarray, words: Sequence[Word], config: PipelineConfig) -> ParsedTable:
    """Parse a table segment image into a grid of (spanning) cells."""
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if config.parser_params.erase_text and words:
        gray = mask_text(gray, words, pad=config.px(4))
    horizontal, vertical = detect_segments(gray, config)
    h_lines, v_lines = build_lines(horizontal, vertical, width, height, config)
    table = build_grid(h_lines, v_lines, width, height, config)
    table.refined_bbox = BBox(table.xs[0], table.ys[0], table.xs[-1], table.ys[-1])
    log.debug("Parsed table %dx%d: %d rows, %d cols, %d cells", width, height, table.n_rows, table.n_cols, len(table.cells))
    return table


def remove_empty_lines(table: ParsedTable, texts: dict[int, str]) -> ParsedTable:
    """Refinement: drop grid rows/columns whose cells are all empty.

    Spurious boundaries (e.g. a double border detected as two lines) create
    empty bands; removing them yields the refined table region. A row/column
    is only removed when *every* cell touching it is empty.
    """
    matrix = table.matrix()
    if not matrix:
        return table
    keep_rows = [r for r in range(table.n_rows) if any(texts.get(cell.id, "").strip() for cell in matrix[r])]
    keep_cols = [c for c in range(table.n_cols) if any(texts.get(matrix[r][c].id, "").strip() for r in range(table.n_rows))]
    if len(keep_rows) == table.n_rows and len(keep_cols) == table.n_cols:
        return table
    if not keep_rows or not keep_cols:
        return ParsedTable(table.width, table.height, [0.0, float(table.width - 1)], [0.0, float(table.height - 1)], [],
                           table.horizontal_lines, table.vertical_lines, table.refined_bbox)

    row_map = {r: i for i, r in enumerate(keep_rows)}
    col_map = {c: i for i, c in enumerate(keep_cols)}
    new_cells: list[GridCell] = []
    for cell in table.cells:
        rows = [row_map[r] for r in range(cell.row, cell.row + cell.row_span) if r in row_map]
        cols = [col_map[c] for c in range(cell.col, cell.col + cell.col_span) if c in col_map]
        if not rows or not cols:
            continue
        new_cells.append(GridCell(
            id=cell.id, row=rows[0], col=cols[0], row_span=rows[-1] - rows[0] + 1,
            col_span=cols[-1] - cols[0] + 1, bbox=cell.bbox, text=cell.text,
        ))
    # Boundaries: keep the left/top boundary of each kept band plus the final edge.
    xs = [table.xs[c] for c in keep_cols] + [table.xs[keep_cols[-1] + 1]]
    ys = [table.ys[r] for r in keep_rows] + [table.ys[keep_rows[-1] + 1]]
    refined = BBox(xs[0], ys[0], xs[-1], ys[-1])
    return ParsedTable(table.width, table.height, xs, ys, new_cells,
                       table.horizontal_lines, table.vertical_lines, refined)
