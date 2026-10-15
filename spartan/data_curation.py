"""Stage 5 - Data curation (digital copy of each table) and text acquisition.

For every parsed table this stage:

1. Fills the cells with text - from the PDF text layer / page OCR by geometric
   assignment, or with the paper's single-/multi-cell OCR strategies.
2. Refines the grid (drops empty rows/columns) and rejects "tables" that are
   only a bordered box of text.
3. Splits the grid at **full-width rows**. Full-width rows at the top are the
   table header (title); a full-width row in the middle starts a new section
   whose title it is; trailing ones become notes.
4. Detects the **table type**:

   * *Key-value*: two columns, a significant width difference and at most five
     rows (paper), or keys that mostly end with ``:`` (extension).
   * *Nested-column*: the first row has fewer cells than the second, checked
     top-down until the number of cells stops changing. Parent and child edges
     align by construction of the grid. Parent and child header texts are joined
     ("Discontinued" + "OPN" -> "Discontinued OPN").
   * *Normal*: everything else.

5. Expands spanning (merged) cells by repeating their value in every row/column
   they span, and records each merged cell with its span and merge type.
6. Applies the column-trend OCR auto-correction to OCR-derived tables.
7. Takes the table title from the preceding text when the table has none, and
   merges similar tables on the same page.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Sequence

from .config import OcrScope, PipelineConfig
from .models import (
    BBox,
    GridCell,
    MergedCellInfo,
    MergeType,
    ParsedTable,
    Segment,
    SegmentKind,
    Table,
    TableSource,
    TableType,
    TextBlock,
    TextSource,
    words_to_text,
)
from .ocr import OcrEngine, correct_ocr_column, ocr_cells
from .table_merging import can_merge, is_repeated_header, merge_tables
from .table_parser import parse_table, remove_empty_lines
from .text_extraction import assign_words_to_cells

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Cell text
# --------------------------------------------------------------------------- #
def fill_cell_text(
    segment: Segment, parsed: ParsedTable, config: PipelineConfig, ocr_engine: OcrEngine | None
) -> tuple[dict[int, str], bool]:
    """Return ``({cell_id: text}, used_ocr)`` for every cell of the table."""
    ocr_words = sum(1 for w in segment.words if w.source == "ocr")
    mostly_ocr = bool(segment.words) and ocr_words > len(segment.words) / 2
    scope = config.ocr_params.scope
    if mostly_ocr and ocr_engine is not None and scope is not OcrScope.PAGE:
        shrink = config.px(config.curation_params.cell_shrink)
        return ocr_cells(segment.image, parsed.cells, ocr_engine, config.ocr_params, shrink), True
    return assign_words_to_cells(parsed.cells, segment.words), mostly_ocr


# --------------------------------------------------------------------------- #
# Structure analysis
# --------------------------------------------------------------------------- #
def _distinct(row: Sequence[GridCell]) -> list[GridCell]:
    seen, out = set(), []
    for cell in row:
        if cell.id not in seen:
            seen.add(cell.id)
            out.append(cell)
    return out


def _is_full_width(row: Sequence[GridCell]) -> bool:
    return len(row) >= 2 and all(c.id == row[0].id for c in row)


def _split_sections(matrix: list[list[GridCell]]) -> list[tuple[list[str], list[int], list[str]]]:
    """Split rows at full-width rows. Returns ``[(title_lines, row_indices, notes)]``."""
    sections: list[tuple[list[str], list[int], list[str]]] = [([], [], [])]
    for r, row in enumerate(matrix):
        if _is_full_width(row):
            text = row[0].text.strip()
            title, rows, notes = sections[-1]
            if not rows:
                if text:
                    title.append(text)
            else:
                sections.append(([text] if text else [], [], []))
        else:
            sections[-1][1].append(r)
    # A trailing section without data rows is a footnote of the previous one.
    cleaned: list[tuple[list[str], list[int], list[str]]] = []
    for title, rows, notes in sections:
        if rows:
            cleaned.append((title, rows, notes))
        elif cleaned:
            cleaned[-1][2].extend(title)
        elif title:
            cleaned.append((title, [], notes))
    return cleaned


def _collapse_columns(matrix: list[list[GridCell]], rows: list[int], widths: list[float]) -> tuple[list[list[GridCell]], list[float]]:
    """Section view with redundant atomic columns removed.

    The atomic grid is shared by the whole table, so lines that exist only in
    one section split columns in every other section. Adjacent atomic columns
    that are covered by the very same cell in every row of the section carry no
    information and are folded into one column (their widths are added).
    """
    n_cols = len(matrix[0]) if matrix else 0
    groups: list[list[int]] = []
    for c in range(n_cols):
        if groups and all(matrix[r][c].id == matrix[r][groups[-1][-1]].id for r in rows):
            groups[-1].append(c)
        else:
            groups.append([c])
    sub = [[matrix[r][g[0]] for g in groups] for r in rows]
    sub_widths = [sum(widths[c] for c in g) for g in groups]
    return sub, sub_widths


def _header_depth(sub: list[list[GridCell]], max_rows: int) -> int:
    """Paper: nested headers have fewer cells in a row than in the row below it."""
    depth = 1
    while depth < len(sub) - 1 and depth < max_rows:
        if len(_distinct(sub[depth])) > len(_distinct(sub[depth - 1])):
            depth += 1
        else:
            break
    return depth


def _column_names(header: list[list[GridCell]]) -> list[str]:
    """Join parent and child header texts top-down ("Discontinued" + "OPN")."""
    names = []
    for c in range(len(header[0]) if header else 0):
        parts: list[str] = []
        last_id = None
        for row in header:
            cell = row[c]
            text = cell.text.strip()
            if cell.id != last_id and text and (not parts or parts[-1] != text):
                parts.append(text)
            last_id = cell.id
        names.append(" ".join(parts))
    return names


_SHAPE_DIGITS = re.compile(r"\d+")
_SHAPE_LETTERS = re.compile(r"[^\W\d_]+")


def _shape(text: str) -> str:
    """Coarse character-class shape: 'MVA-2000+' -> 'A-9+'."""
    return _SHAPE_LETTERS.sub("A", _SHAPE_DIGITS.sub("9", re.sub(r"\s+", " ", text.strip())))


def _first_row_is_data(first: list[str], data: list[list[str]]) -> bool:
    """True when the first row has the same shape as the data below it (no header row)."""
    votes = total = 0
    for c, head in enumerate(first):
        values = [row[c] for row in data if c < len(row) and row[c].strip()]
        if not head.strip() or not values:
            continue
        dominant, count = Counter(_shape(v) for v in values).most_common(1)[0]
        if count / len(values) < 0.5:
            continue
        total += 1
        votes += _shape(head) == dominant
    return total > 0 and votes / total >= 0.5


def _is_code(text: str) -> bool:
    """Part-number-like token: no spaces, letters and digits mixed."""
    t = text.strip()
    return len(t) >= 4 and " " not in t and any(c.isdigit() for c in t) and any(c.isalpha() for c in t)


def _is_key_value(sub: list[list[GridCell]], widths: list[float], config: PipelineConfig) -> bool:
    """Paper: two columns, significant width difference, at most five rows.

    Extensions: keys mostly ending with ':' also qualify; a first column made of
    part numbers (below the first row) is a data column, not keys.
    """
    p = config.curation_params
    if len(widths) != 2 or not sub:
        return False
    if any(len(_distinct(row)) != 2 for row in sub):
        return False
    keys = [row[0].text.strip() for row in sub if row[0].text.strip()]
    if not keys:
        return False
    colon_ratio = sum(k.endswith(":") for k in keys) / len(keys)
    if colon_ratio >= p.kv_colon_ratio:
        return True
    body = keys[1:]
    if body and sum(_is_code(k) for k in body) / len(body) >= 0.6:
        return False
    ratio = max(widths) / max(min(widths), 1.0)
    return ratio >= p.kv_width_ratio and len(sub) <= p.kv_max_rows


def _form_pairs(sub: list[list[GridCell]]) -> list[list[str]] | None:
    """Read an irregular *form* grid (label | value cells) as key/value pairs.

    Two layouts are recognised:

    * every row holds at most two non-empty cells (label, value);
    * labels end with ':' and each label is directly followed by its value
      ("PDN Number: | 2023... | PDN Date: | July 07, 2023").
    """
    rows = [[c.text.strip() for c in _distinct(row) if c.text.strip()] for row in sub]
    rows = [r for r in rows if r]
    if len(rows) < 2:
        return None
    if all(len(r) <= 2 for r in rows) and sum(len(r) == 2 for r in rows) >= len(rows) / 2:
        return [[r[0], r[1] if len(r) > 1 else ""] for r in rows]

    cells = [t for r in rows for t in r]
    labels = [t for t in cells if t.endswith(":")]
    if len(labels) < 0.3 * len(cells):
        return None
    pairs: list[list[str]] = []
    for r in rows:
        i = 0
        while i < len(r):
            if r[i].endswith(":"):
                if i + 1 >= len(r) or r[i + 1].endswith(":"):
                    return None          # a label row above a value row: header/data, not a form
                pairs.append([r[i], r[i + 1]])
                i += 2
            else:
                pairs.append([r[i], ""])
                i += 1
    return pairs


def _clean_pairs(pairs: list[list[str]]) -> list[list[str]]:
    """Normalise key/value rows: ': value' -> 'value'."""
    return [[k.strip(), re.sub(r"^\s*:\s*", "", v).strip()] for k, v, *_ in (p + [""] for p in pairs)]


def build_tables(
    parsed: ParsedTable,
    texts: dict[int, str],
    segment: Segment,
    page_number: int,
    page_offset: BBox,
    config: PipelineConfig,
    text_source: str,
    used_ocr: bool,
) -> list[Table]:
    """Turn a parsed grid with cell texts into one or more curated tables."""
    for cell in parsed.cells:
        cell.text = texts.get(cell.id, "").strip()
    parsed = remove_empty_lines(parsed, texts)
    if parsed.n_rows == 0 or parsed.n_cols == 0:
        return []
    if parsed.n_cols == 1 and parsed.n_rows <= 2:
        return []                   # a bordered box (optionally with a title): text, not a table
    if len({c.id for c in parsed.cells}) <= 1:
        return []

    matrix = parsed.matrix()
    atomic_widths = [b - a for a, b in zip(parsed.xs[:-1], parsed.xs[1:])]
    source_box = segment.bbox.translate(page_offset.x0, page_offset.y0)
    tables: list[Table] = []

    sections = _split_sections(matrix) if parsed.n_cols >= 2 else [([], list(range(parsed.n_rows)), [])]
    for title_lines, rows, notes in sections:
        if not rows:
            continue
        sub, widths = _collapse_columns(matrix, rows, atomic_widths)
        n_cols = len(widths)
        if n_cols == 1 and len(sub) <= 2:
            continue
        title = " ".join(title_lines).strip()

        form = _form_pairs(sub) if n_cols > 2 else None
        if form is not None:
            tables.append(Table(
                type=TableType.KEY_VALUE, columns=["key", "value"], rows=_clean_pairs(form),
                column_widths=[], title=title, title_source="table" if title else "none",
                header_rows=0, notes=list(notes), detection_method=segment.detection_method or "boundary",
                text_source=TextSource.OCR.value if used_ocr else text_source,
                sources=[TableSource(page_number, segment.id, source_box)],
            ))
            continue

        if _is_key_value(sub, widths, config):
            ttype, header_rows, raw_header = TableType.KEY_VALUE, 0, []
            columns = ["key", "value"]
        else:
            depth = _header_depth(sub, config.curation_params.max_header_rows) if len(sub) > 1 else 1
            ttype = TableType.NESTED if depth > 1 else TableType.NORMAL
            header_rows = depth
            raw_header = _column_names(sub[:depth])
            if depth == 1 and len(sub) > 1 and _first_row_is_data(
                [c.text for c in sub[0]], [[c.text for c in row] for row in sub[1:]]
            ):
                header_rows, raw_header = 0, []          # header-less list table
            columns = [name or f"column_{i + 1}" for i, name in enumerate(raw_header)] or [
                f"column_{i + 1}" for i in range(n_cols)
            ]

        values: list[list[str]] = []
        merged: list[MergedCellInfo] = []
        seen_cells: set[int] = set()
        data = sub[header_rows:]
        for i, row in enumerate(data):
            row_vals = [cell.text for cell in row]
            if not any(v.strip() for v in row_vals):
                continue
            if raw_header and is_repeated_header(row_vals, raw_header, config.curation_params):
                continue            # header row repeated inside the table
            values.append(row_vals)
            for c, cell in enumerate(row):
                if cell.id in seen_cells or cell.merge_type is MergeType.NONE:
                    continue
                seen_cells.add(cell.id)
                row_span = sum(1 for later in data[i:] if any(o.id == cell.id for o in later))
                col_span = sum(1 for other in row if other.id == cell.id)
                merge_type = (
                    MergeType.BOTH if row_span > 1 and col_span > 1
                    else MergeType.ROW if row_span > 1
                    else MergeType.COLUMN if col_span > 1
                    else MergeType.NONE
                )
                if merge_type is not MergeType.NONE:
                    merged.append(MergedCellInfo(len(values) - 1, c, row_span, col_span, merge_type, cell.text))

        if used_ocr:
            for c in range(n_cols):
                fixed = correct_ocr_column([row[c] for row in values])
                for row, val in zip(values, fixed):
                    row[c] = val
        if ttype is TableType.KEY_VALUE:
            values = _clean_pairs(values)

        tables.append(Table(
            type=ttype,
            columns=columns,
            rows=values,
            column_widths=widths,
            title=title,
            title_source="table" if title else "none",
            header_rows=header_rows,
            merged_cells=merged,
            notes=list(notes),
            detection_method=segment.detection_method or "boundary",
            text_source=TextSource.OCR.value if used_ocr else text_source,
            sources=[TableSource(page_number, segment.id, source_box)],
            raw_header=raw_header,
        ))
    return tables




# --------------------------------------------------------------------------- #
# Page-level curation
# --------------------------------------------------------------------------- #
def _context_title(text: str, max_chars: int) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""
    candidate = lines[-1]
    return candidate if len(candidate) <= max_chars else ""


def curate_segments(
    segments: Sequence[Segment],
    page_number: int,
    page_offset: BBox,
    config: PipelineConfig,
    text_source: TextSource,
    ocr_engine: OcrEngine | None,
    debug: dict | None = None,
) -> list[tuple[str, object]]:
    """Curate all segments of one region into ordered ``("text"|"table", obj)`` items."""
    items: list[tuple[str, object]] = []
    for seg in segments:
        page_box = seg.bbox.translate(page_offset.x0, page_offset.y0)
        if seg.kind is not SegmentKind.TABLE:
            text = words_to_text(seg.words)
            if text.strip():
                items.append(("text", TextBlock(seg.id, seg.kind, text, page_box)))
            continue

        parsed = parse_table(seg.image, seg.words, config)
        texts, used_ocr = fill_cell_text(seg, parsed, config, ocr_engine)
        tables = build_tables(parsed, texts, seg, page_number, page_offset, config, text_source.value, used_ocr)
        if debug is not None:
            debug.setdefault("parsed", []).append((seg, parsed, tables))
        if not tables:
            text = words_to_text(seg.words)
            if text.strip():
                items.append(("text", TextBlock(seg.id, SegmentKind.BOXED_TEXT, text, page_box)))
            continue
        for table in tables:
            items.append(("table", table))

    # Titles from the preceding text segment (pre-text / inter-text), as in the paper.
    max_chars = config.curation_params.context_title_max_chars
    for i, (kind, obj) in enumerate(items):
        if kind == "table" and not obj.title and i > 0 and items[i - 1][0] == "text":
            title = _context_title(items[i - 1][1].text, max_chars)
            if title:
                obj.title, obj.title_source = title, "context"

    return merge_page_tables(items, config)


def merge_page_tables(items: list[tuple[str, object]], config: PipelineConfig) -> list[tuple[str, object]]:
    """Merge consecutive similar tables on the same page (paper, data-curation step)."""
    merged: list[tuple[str, object]] = []
    for kind, obj in items:
        if kind == "table" and merged and merged[-1][0] == "table" and can_merge(
            merged[-1][1], obj, config.curation_params, config.scale, same_page=True
        ):
            merge_tables(merged[-1][1], obj, config.curation_params)
            log.debug("Merged two similar tables on the same page")
            continue
        merged.append((kind, obj))
    return merged
