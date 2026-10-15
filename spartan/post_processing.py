"""Stage 6 - Post-processing: from page results to one document.

* **Repeated page headers/footers** are detected line by line: a text line that
  recurs on two or more pages (ignoring digits, so "Page 1 of 3" matches
  "Page 2 of 3") is boilerplate. Boilerplate is kept once in the document text
  and does not block table continuation.
* **Tables continuing across pages** are merged: a table that ends a page and a
  table that starts the next page (with only boilerplate or very short text in
  between) are merged when their column count, column widths, header rows and
  titles are compatible. Repeated header rows are dropped, and a continuation
  without a header keeps its first row as data. ``continued_from_previous`` /
  ``continued_to_next`` are set on the per-page tables.
"""

from __future__ import annotations

import copy
import logging
import re
from collections import defaultdict
from typing import Any, Sequence

from .config import PipelineConfig
from .models import PageResult, Table, TextBlock
from .table_merging import can_merge, merge_tables

log = logging.getLogger(__name__)

_INSIGNIFICANT_CHARS = 40


def _fingerprint(line: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\d+", "#", line.lower())).strip()


def _lines(text: str) -> list[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def find_boilerplate_lines(pages: Sequence[PageResult]) -> set[str]:
    """Fingerprints of text lines that appear on two or more pages."""
    if len(pages) < 2:
        return set()
    seen: dict[str, set[int]] = defaultdict(set)
    for page in pages:
        for block in page.text_blocks:
            for line in _lines(block.text):
                seen[_fingerprint(line)].add(page.page_number)
    return {fp for fp, pages_with in seen.items() if len(pages_with) >= 2 and len(fp) >= 3}


def _significant_text(block: TextBlock, boilerplate: set[str]) -> str:
    return " ".join(ln for ln in _lines(block.text) if _fingerprint(ln) not in boilerplate)


def _is_ignorable(block: TextBlock, boilerplate: set[str]) -> bool:
    return len(_significant_text(block, boilerplate)) < _INSIGNIFICANT_CHARS


def merge_document_tables(pages: Sequence[PageResult], boilerplate: set[str], config: PipelineConfig) -> list[Table]:
    """Merge tables that continue across page breaks. Returns the document's tables."""
    tables: list[Table] = []          # merged, document-level tables
    current: Table | None = None      # document table being extended
    last_piece: Table | None = None   # per-page table appended last
    last_page = -1
    blocked = False                   # significant text seen since the last table

    for page in pages:
        for kind, obj in page.items:
            if kind == "text":
                if current is not None and not _is_ignorable(obj, boilerplate):
                    blocked = True
                continue
            if (
                current is not None
                and last_piece is not None
                and not blocked
                and page.page_number == last_page + 1
                and can_merge(current, obj, config.curation_params, config.scale, same_page=False)
            ):
                log.info("Table on page %d continues the table from page %d", page.page_number, last_page)
                merge_tables(current, copy.deepcopy(obj), config.curation_params)
                last_piece.continued_to_next = True
                obj.continued_from_previous = True
            else:
                current = copy.deepcopy(obj)
                tables.append(current)
            last_piece = obj
            last_page = page.page_number
            blocked = False

    for table in tables:              # document-level tables are complete
        table.continued_from_previous = False
        table.continued_to_next = False
    return _deduplicate(tables)


def _deduplicate(tables: list[Table]) -> list[Table]:
    """Drop tables repeated verbatim on several pages (e.g. a box in every page header)."""
    unique: list[Table] = []
    for table in tables:
        twin = next((u for u in unique if u.columns == table.columns and u.rows == table.rows), None)
        if twin is None:
            unique.append(table)
        else:
            twin.sources.extend(table.sources)
            log.debug("Dropped a table repeated on page(s) %s", [s.page for s in table.sources])
    return unique


def build_document(document_name: str, pages: Sequence[PageResult], config: PipelineConfig) -> dict[str, Any]:
    """Assemble the final document dictionary."""
    boilerplate = find_boilerplate_lines(pages)
    tables = merge_document_tables(pages, boilerplate, config)

    text_items = []
    emitted_boilerplate: set[str] = set()
    for page in pages:
        for block in page.text_blocks:
            kept = []
            for line in _lines(block.text):
                fp = _fingerprint(line)
                if fp in boilerplate:
                    if fp in emitted_boilerplate:
                        continue          # keep the first occurrence of a header/footer only
                    emitted_boilerplate.add(fp)
                kept.append(line)
            if not kept:
                continue
            text_items.append({
                "page": page.page_number,
                "id": block.id,
                "kind": block.kind.value,
                "text": "\n".join(kept),
                "boilerplate": all(_fingerprint(ln) in boilerplate for ln in kept),
            })

    table_dicts = []
    for i, table in enumerate(tables, start=1):
        table_dicts.append({"id": f"table_{i}", "pages": sorted({s.page for s in table.sources}), **table.to_dict()})

    return {
        "document_name": document_name,
        "tables": table_dicts,
        "text": text_items,
        "full_text": "\n\n".join(t["text"] for t in text_items),
        "pages": [p.to_dict() for p in pages],
        "_tables": tables,        # live objects for exporters/plugins (stripped before JSON output)
    }
