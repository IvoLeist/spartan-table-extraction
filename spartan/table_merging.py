"""Merging of similar / continuing tables (used per page and across pages).

Two tables are merged when they have the same number of columns, their column
widths agree within a tolerance, and their titles do not conflict. The second
table's header row is then handled in one of two ways:

* it repeats the first table's header (fuzzy match) -> it is dropped;
* it does not match -> the "header" was actually a data row of a continuation
  table without a repeated header, so it is prepended to the data rows.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Sequence

from .config import CurationParams
from .models import Table, TableType


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^0-9a-z]+", " ", (text or "").lower())).strip()


def similar(a: str, b: str, threshold: float) -> bool:
    na, nb = normalize_text(a), normalize_text(b)
    if na == nb:
        return True
    if not na or not nb:
        return False
    return SequenceMatcher(None, na, nb).ratio() >= threshold


def headers_match(a: Sequence[str], b: Sequence[str], threshold: float) -> bool:
    if len(a) != len(b):
        return False
    return all(similar(x, y, threshold) for x, y in zip(a, b))


def is_repeated_header(row: Sequence[str], header: Sequence[str], params: CurationParams) -> bool:
    """True when a data row repeats the header (fuzzy, ignoring empty header cells)."""
    pairs = [(r, h) for r, h in zip(row, header) if h.strip()]
    return len(row) == len(header) and bool(pairs) and all(similar(r, h, params.header_similarity) for r, h in pairs)


def widths_match(a: Sequence[float], b: Sequence[float], params: CurationParams, px_scale: float) -> bool:
    if len(a) != len(b) or not a:
        return False
    total = max(sum(a), sum(b), 1.0)
    tol = max(params.merge_width_tolerance * px_scale, params.merge_width_tolerance_ratio * total)
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def same_title(a: str, b: str) -> bool:
    """Titles must match exactly (after normalisation, ignoring "continued").

    Fuzzy matching is deliberately not used: "EOL devices with LTB" and
    "EOL devices with no LTB" are different tables.
    """
    def clean(t: str) -> list[str]:
        return re.sub(r"\b(continued|cont d|contd|cont)\b", " ", normalize_text(t)).split()

    return clean(a) == clean(b)


def titles_compatible(first: Table, second: Table, params: CurationParams, strict: bool) -> bool:
    """``strict``: any two differing titles conflict; otherwise only table-sourced titles do."""
    if not first.title or not second.title:
        return True
    if same_title(first.title, second.title):
        return True
    if strict:
        return False
    return not (first.title_source == "table" and second.title_source == "table")


def can_merge(first: Table, second: Table, params: CurationParams, px_scale: float, *, same_page: bool) -> bool:
    if first.type is TableType.KEY_VALUE or second.type is TableType.KEY_VALUE:
        return False
    if first.n_cols != second.n_cols or first.n_cols < 2:
        return False
    if not titles_compatible(first, second, params, strict=same_page):
        return False
    same_header = headers_match(first.raw_header, second.raw_header, params.header_similarity)
    if same_page:
        # Paper: same column count and widths, or identical (non-empty) table headers.
        # Paper: same column count and widths, or identical (non-empty) table headers.
        # A header-less block right below a table with the same widths continues it
        # (tables drawn as several boxes).
        titled_same = bool(first.title) and same_title(first.title, second.title)
        same_widths = widths_match(first.column_widths, second.column_widths, params, px_scale)
        return (same_widths and (same_header or second.header_rows == 0)) or (titled_same and same_header)
    return same_header or widths_match(first.column_widths, second.column_widths, params, px_scale)


def merge_tables(first: Table, second: Table, params: CurationParams) -> Table:
    """Append ``second`` to ``first`` (in place) and return ``first``."""
    if headers_match(first.raw_header, second.raw_header, params.header_similarity):
        extra_rows = second.rows
    elif second.header_rows == 1 and second.raw_header:
        extra_rows = [list(second.raw_header)] + second.rows
    else:
        extra_rows = second.rows
    # Header rows repeated in the middle of the continuation are dropped.
    kept_idx = [i for i, row in enumerate(extra_rows) if not is_repeated_header(row, first.raw_header, params)]
    shift = len(extra_rows) - len(second.rows)          # 1 if the header row became data
    new_index = {old: new for new, old in enumerate(kept_idx)}
    offset = len(first.rows)
    first.rows.extend(extra_rows[i] for i in kept_idx)
    for m in second.merged_cells:
        if m.row + shift in new_index:
            m.row = offset + new_index[m.row + shift]
            first.merged_cells.append(m)
    if not first.title and second.title:
        first.title, first.title_source = second.title, second.title_source
    first.notes.extend(second.notes)
    first.sources.extend(second.sources)
    first.column_widths = [(a + b) / 2 for a, b in zip(first.column_widths, second.column_widths)]
    first.continued_to_next = second.continued_to_next
    return first
