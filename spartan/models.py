"""Core data structures shared by every pipeline stage.

All geometry is expressed in *pixel* coordinates of the image the object belongs
to, with the origin at the top-left corner, ``x`` growing to the right and ``y``
growing downwards. Boxes are half-open ``(x0, y0, x1, y1)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class BBox:
    """Axis-aligned bounding box ``(x0, y0, x1, y1)``."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    def intersects(self, other: "BBox") -> bool:
        return self.x0 < other.x1 and other.x0 < self.x1 and self.y0 < other.y1 and other.y0 < self.y1

    def intersection(self, other: "BBox") -> "BBox | None":
        x0, y0 = max(self.x0, other.x0), max(self.y0, other.y0)
        x1, y1 = min(self.x1, other.x1), min(self.y1, other.y1)
        if x0 >= x1 or y0 >= y1:
            return None
        return BBox(x0, y0, x1, y1)

    def overlap_ratio(self, other: "BBox") -> float:
        """Intersection area divided by the area of the *smaller* box."""
        inter = self.intersection(other)
        if inter is None:
            return 0.0
        smaller = min(self.area, other.area)
        return inter.area / smaller if smaller > 0 else 0.0

    def contains_point(self, x: float, y: float) -> bool:
        return self.x0 <= x < self.x1 and self.y0 <= y < self.y1

    def contains(self, other: "BBox", tolerance: float = 0.0) -> bool:
        return (
            other.x0 >= self.x0 - tolerance
            and other.y0 >= self.y0 - tolerance
            and other.x1 <= self.x1 + tolerance
            and other.y1 <= self.y1 + tolerance
        )

    def translate(self, dx: float, dy: float) -> "BBox":
        return BBox(self.x0 + dx, self.y0 + dy, self.x1 + dx, self.y1 + dy)

    def expand(self, margin: float) -> "BBox":
        return BBox(self.x0 - margin, self.y0 - margin, self.x1 + margin, self.y1 + margin)

    def clip(self, width: float, height: float) -> "BBox":
        return BBox(max(0, self.x0), max(0, self.y0), min(width, self.x1), min(height, self.y1))

    def union(self, other: "BBox") -> "BBox":
        return BBox(min(self.x0, other.x0), min(self.y0, other.y0), max(self.x1, other.x1), max(self.y1, other.y1))

    def to_int(self) -> "BBox":
        return BBox(int(round(self.x0)), int(round(self.y0)), int(round(self.x1)), int(round(self.y1)))

    def as_slice(self) -> tuple[slice, slice]:
        """Return ``(rows, cols)`` slices for indexing a NumPy image."""
        b = self.to_int()
        return slice(int(b.y0), int(b.y1)), slice(int(b.x0), int(b.x1))

    def to_list(self, ndigits: int = 1) -> list[float]:
        return [round(self.x0, ndigits), round(self.y0, ndigits), round(self.x1, ndigits), round(self.y1, ndigits)]

    @staticmethod
    def union_all(boxes: Iterable["BBox"]) -> "BBox":
        boxes = list(boxes)
        if not boxes:
            raise ValueError("union_all() requires at least one box")
        return BBox(
            min(b.x0 for b in boxes),
            min(b.y0 for b in boxes),
            max(b.x1 for b in boxes),
            max(b.y1 for b in boxes),
        )

    @staticmethod
    def from_xywh(x: float, y: float, w: float, h: float) -> "BBox":
        return BBox(x, y, x + w, y + h)


# --------------------------------------------------------------------------- #
# Text
# --------------------------------------------------------------------------- #
class TextSource(str, Enum):
    PDF = "pdf_text_layer"
    OCR = "ocr"
    HYBRID = "pdf_text_layer+ocr"
    NONE = "none"


@dataclass(slots=True)
class Word:
    """A single word with its pixel bounding box."""

    text: str
    bbox: BBox                   # layout box (font ascent..descent), used for line grouping
    source: str = "pdf"          # "pdf" or "ocr"
    confidence: float = 1.0
    ink_bbox: BBox | None = None # tight glyph box, used to mask text without touching rules
    vertical: bool = False       # runs top-to-bottom on the rendered page (margin notes, rotated labels)

    @property
    def ink(self) -> BBox:
        return self.ink_bbox or self.bbox

    def translate(self, dx: float, dy: float) -> "Word":
        ink = self.ink_bbox.translate(dx, dy) if self.ink_bbox else None
        return Word(self.text, self.bbox.translate(dx, dy), self.source, self.confidence, ink, self.vertical)


def words_in_box(words: Sequence[Word], box: BBox, *, relative: bool = True) -> list[Word]:
    """Words whose centre lies inside ``box``; optionally re-expressed relative to it."""
    selected = [w for w in words if box.contains_point(w.bbox.cx, w.bbox.cy)]
    if relative:
        selected = [w.translate(-box.x0, -box.y0) for w in selected]
    return selected


def group_words_into_lines(words: Sequence[Word], tolerance_ratio: float = 0.5) -> list[list[Word]]:
    """Cluster words into visual text lines (top-to-bottom, each line left-to-right).

    Two words belong to the same line when their vertical centres differ by less
    than ``tolerance_ratio`` times the smaller word height.
    """
    if not words:
        return []
    ordered = sorted(words, key=lambda w: (w.bbox.cy, w.bbox.x0))
    lines: list[list[Word]] = []
    line_cy: list[float] = []
    line_h: list[float] = []
    for word in ordered:
        placed = False
        # Only the last few lines can match because words are sorted by y.
        for idx in range(len(lines) - 1, max(-1, len(lines) - 4), -1):
            tol = tolerance_ratio * min(line_h[idx], max(word.bbox.height, 1.0))
            if abs(word.bbox.cy - line_cy[idx]) <= tol:
                lines[idx].append(word)
                n = len(lines[idx])
                line_cy[idx] = (line_cy[idx] * (n - 1) + word.bbox.cy) / n
                line_h[idx] = max(line_h[idx], word.bbox.height)
                placed = True
                break
        if not placed:
            lines.append([word])
            line_cy.append(word.bbox.cy)
            line_h.append(max(word.bbox.height, 1.0))
    for line in lines:
        line.sort(key=lambda w: w.bbox.x0)
    lines.sort(key=lambda ln: min(w.bbox.y0 for w in ln))
    return lines


def words_to_text(words: Sequence[Word], line_separator: str = "\n") -> str:
    """Reconstruct reading-order text from positioned words."""
    return line_separator.join(" ".join(w.text for w in line) for line in group_words_into_lines(words))


# --------------------------------------------------------------------------- #
# Images
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class PageImage:
    """A rendered page plus the derived images later stages reuse."""

    page_number: int                 # 1-based
    bgr: np.ndarray                  # original render (H x W x 3, uint8)
    gray: np.ndarray                 # grayscale (H x W, uint8)
    binary: np.ndarray               # ink mask: 255 = ink, 0 = background
    dpi: int
    width_pt: float = 0.0
    height_pt: float = 0.0

    @property
    def height(self) -> int:
        return int(self.bgr.shape[0])

    @property
    def width(self) -> int:
        return int(self.bgr.shape[1])


@dataclass(slots=True)
class ImageRegion:
    """A rectangular part of a page (whole page or a detected column)."""

    index: int
    bbox: BBox                       # in page coordinates
    image: np.ndarray                # BGR crop
    words: list[Word]                # in region coordinates


# --------------------------------------------------------------------------- #
# Region segmentation
# --------------------------------------------------------------------------- #
class SegmentKind(str, Enum):
    PRE_TEXT = "pre_text"
    TABLE = "table"
    INTER_TEXT = "inter_text"
    POST_TEXT = "post_text"
    BOXED_TEXT = "boxed_text"   # a bordered box that turned out not to be a table


@dataclass(slots=True)
class TableCandidate:
    """A table region found by one of the region-segmentation detectors."""

    bbox: BBox
    method: str                      # "boundary" or "text"
    # For text-based (borderless) tables: the synthetic grid used to draw pseudo boundaries.
    pseudo_columns: list[float] = field(default_factory=list)  # x separators (region coords)
    pseudo_rows: list[float] = field(default_factory=list)     # y separators (region coords)


@dataclass(slots=True)
class Segment:
    """A vertical slice of a region: text before/between/after tables, or a table."""

    id: str
    kind: SegmentKind
    bbox: BBox                       # in region coordinates
    image: np.ndarray
    words: list[Word]                # in segment coordinates
    detection_method: str | None = None   # tables only
    pseudo_bordered: bool = False         # tables only: pseudo boundaries were drawn
    pseudo_columns: list[float] = field(default_factory=list)  # pseudo separators (segment coords)
    pseudo_rows: list[float] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Table parsing
# --------------------------------------------------------------------------- #
class MergeType(str, Enum):
    NONE = "none"
    ROW = "row"
    COLUMN = "column"
    BOTH = "both"


@dataclass(slots=True)
class GridCell:
    """A (possibly spanning) cell of the parsed grid."""

    id: int
    row: int
    col: int
    row_span: int
    col_span: int
    bbox: BBox                       # in table-image coordinates
    text: str = ""

    @property
    def merge_type(self) -> MergeType:
        if self.row_span > 1 and self.col_span > 1:
            return MergeType.BOTH
        if self.row_span > 1:
            return MergeType.ROW
        if self.col_span > 1:
            return MergeType.COLUMN
        return MergeType.NONE


@dataclass(slots=True)
class ParsedTable:
    """Output of the table parser: a grid of atomic rows/columns plus spanning cells."""

    width: int
    height: int
    xs: list[float]                  # column boundary x coordinates (len = n_cols + 1)
    ys: list[float]                  # row boundary y coordinates (len = n_rows + 1)
    cells: list[GridCell]
    horizontal_lines: list[tuple[float, float, float, float]] = field(default_factory=list)
    vertical_lines: list[tuple[float, float, float, float]] = field(default_factory=list)
    refined_bbox: BBox | None = None

    @property
    def n_rows(self) -> int:
        return max(0, len(self.ys) - 1)

    @property
    def n_cols(self) -> int:
        return max(0, len(self.xs) - 1)

    @property
    def header_candidates(self) -> list[GridCell]:
        """Cells spanning the full table width (paper: table-header candidates)."""
        return [c for c in self.cells if c.col == 0 and c.col_span == self.n_cols and self.n_cols > 1]

    def matrix(self) -> list[list[GridCell]]:
        """Dense ``n_rows x n_cols`` view; spanning cells are repeated."""
        grid: list[list[GridCell | None]] = [[None] * self.n_cols for _ in range(self.n_rows)]
        for cell in self.cells:
            for r in range(cell.row, cell.row + cell.row_span):
                for c in range(cell.col, cell.col + cell.col_span):
                    grid[r][c] = cell
        return grid  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# Curated tables and results
# --------------------------------------------------------------------------- #
class TableType(str, Enum):
    NORMAL = "normal"
    NESTED = "nested"
    KEY_VALUE = "key_value"


@dataclass(slots=True)
class TableSource:
    page: int
    segment: str
    bbox: BBox                        # in page coordinates

    def to_dict(self) -> dict[str, Any]:
        return {"page": self.page, "segment": self.segment, "bbox": self.bbox.to_list()}


@dataclass(slots=True)
class MergedCellInfo:
    row: int                          # data-row index (0-based, header rows excluded)
    col: int
    row_span: int
    col_span: int
    merge_type: MergeType
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "row": self.row,
            "col": self.col,
            "row_span": self.row_span,
            "col_span": self.col_span,
            "merge_type": self.merge_type.value,
            "text": self.text,
        }


@dataclass(slots=True)
class Table:
    """A fully curated table, ready for JSON/CSV export."""

    type: TableType
    columns: list[str]
    rows: list[list[str]]
    column_widths: list[float] = field(default_factory=list)
    title: str = ""
    title_source: str = "none"        # "table", "context" or "none"
    header_rows: int = 1
    header_from_data: bool = False    # True when no header row was found and names are generic
    merged_cells: list[MergedCellInfo] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    detection_method: str = "boundary"
    text_source: str = TextSource.PDF.value
    continued_from_previous: bool = False
    continued_to_next: bool = False
    sources: list[TableSource] = field(default_factory=list)
    raw_header: list[str] = field(default_factory=list)   # header texts before fallback naming
    header: list[list[str]] = field(default_factory=list)  # header rows as laid out (spanning text repeated)

    @property
    def n_cols(self) -> int:
        return len(self.columns)

    def pairs(self) -> list[dict[str, str]]:
        """Key/value view for key-value tables (first column = key, rest = value)."""
        out = []
        for row in self.rows:
            if not row:
                continue
            out.append({"key": row[0], "value": " ".join(v for v in row[1:] if v).strip()})
        return out

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "type": self.type.value,
            "title": self.title,
            "title_source": self.title_source,
            "columns": self.columns,
            "header": self.header,
            "header_rows": self.header_rows,
            "rows": self.rows,
            "n_rows": len(self.rows),
            "n_cols": self.n_cols,
            "column_widths": [round(w, 1) for w in self.column_widths],
            "merged_cells": [m.to_dict() for m in self.merged_cells],
            "detection_method": self.detection_method,
            "text_source": self.text_source,
            "continued_from_previous": self.continued_from_previous,
            "continued_to_next": self.continued_to_next,
            "sources": [s.to_dict() for s in self.sources],
        }
        if self.type is TableType.KEY_VALUE:
            data["pairs"] = self.pairs()
        if self.notes:
            data["notes"] = self.notes
        return data


@dataclass(slots=True)
class TextBlock:
    id: str
    kind: SegmentKind
    text: str
    bbox: BBox                        # in page coordinates

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind.value, "text": self.text, "bbox": self.bbox.to_list()}


@dataclass(slots=True)
class PageResult:
    page_number: int
    width: int
    height: int
    text_source: TextSource
    regions: list[BBox] = field(default_factory=list)
    # Ordered list of ("text", TextBlock) / ("table", Table) items in reading order.
    items: list[tuple[str, Any]] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def tables(self) -> list[Table]:
        return [obj for kind, obj in self.items if kind == "table"]

    @property
    def text_blocks(self) -> list[TextBlock]:
        return [obj for kind, obj in self.items if kind == "text"]

    def to_dict(self) -> dict[str, Any]:
        segments = []
        for kind, obj in self.items:
            if kind == "table":
                segments.append({"kind": "table", "table": obj.to_dict()})
            else:
                segments.append(obj.to_dict())
        return {
            "page_number": self.page_number,
            "width": self.width,
            "height": self.height,
            "text_source": self.text_source.value,
            "regions": [r.to_list() for r in self.regions],
            "segments": segments,
            "timings": {k: round(v, 3) for k, v in self.timings.items()},
        }
