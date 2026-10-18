# Changelog

## 2.2.0 — 2026-10-18

Measured on the ICDAR 2013 table competition dataset (corrected version, 67 PDFs):
GriTS-Con 77.5 → 81.6, table precision 79 % → 83 %, tables detected 88 % → 89 %,
adjacency F1 81.9 → 85.9. Output on PCN/PDN notices unchanged.

Output:

- New `header` field: the header rows as laid out in the table (spanning text repeated),
  next to the flattened `columns` names. Multi-row headers keep their structure and empty
  header cells stay empty instead of becoming `column_N`.

Text layer:

- Rotated text with pypdfium2 4.x: pdfium returned empty glyph boxes, so landscape pages
  (`/Rotate 90`) lost all their text. The box is now rebuilt from the glyph's matrix.
- Vertical text (margin notes such as the Federal Register side line) is flagged and no
  longer used to detect borderless tables.

Tables:

- Narrow sliver columns made by the edge of a shaded header are folded into their neighbour.
- Columns holding only currency symbols ("$") are joined with the amounts on their right.
- Sections of a financial statement (numeric columns aligned, short headings between) are
  joined into one borderless table.
- Row labels with several indentation levels stay in the borderless table when their values
  line up.
- Rejected as tables: chart axis labels (one value per row) and multi-column prose next to
  an empty column.

## 2.1.0 — 2026-10-18

Measured on the ICDAR 2013 table competition dataset (corrected version, 67 PDFs):
GriTS-Con 69.2 → 77.5, table precision 50 % → 79 %, tables detected 76 % → 88 %,
adjacency F1 71.2 → 81.9. Output on PCN/PDN notices unchanged.

Tables that rule only part of their grid:

- Body rows without row lines (all values of a column in one tall cell) are split at the
  gaps between text lines when the lines look like data (aligned across columns, short,
  with a numeric column); header rows and wrapped sentences are left alone.
- Body rows without column lines (each row one wide cell) are split at the header's column
  boundaries when at least three rows fit them cleanly; section-title rows keep spanning.
- A ruled header box above an unruled body is joined with the body found by the
  borderless detector, instead of the body being discarded.
- Tables placed side by side are separated instead of being parsed as one grid.

Borderless (text-based) tables:

- The grid of a borderless table is built from its known pseudo separators instead of being
  re-detected in the image; text masking used to erase parts of the pseudo lines, which split
  tables at random rows.
- Stacked pieces of one borderless table (split by a section heading or a column starting
  lower down) are joined, unless a caption lies between them or both start with the same row
  label (two tables with the same layout).
- Numeric columns closer together than the phrase gap are split at gaps free in every row.
- Rejected as tables: running text in every column (multi-column page layout), lists with
  symbol-font bullets, rotated chart labels, single rows.

## 2.0.0 — 2026-10-15

Complete rewrite of the original research scripts as the `spartan` package.

- Text comes from the PDF text layer when present; OCR is used only for scanned pages or
  regions without text (`--text-source auto|pdf|ocr`).
- All pipeline stages from the paper implemented and connected: pre-processing, column
  detection, boundary and borderless table detection with pseudo boundaries, LSD-based
  table parsing with spanning cells, data curation (normal / nested / key-value tables),
  cross-page merging and header/footer de-duplication.
- Paper's OCR strategies (`--ocr-scope single|multi|auto`, `--chunk-size`) with the
  row-separator sanity check; pluggable OCR engines (Tesseract, Azure, or your own).
- PCN schema mapping is an optional plugin (`--pcn`): deterministic matching first, then a
  local LLM through Ollama or Azure OpenAI for the remaining fields.
- Installable library (`pip install spartan-tables`, imported as `spartan`) with optional
  extras; works with OpenCV 4 and 5; type hints shipped (`py.typed`).
- CLI, Python API and FastAPI service; JSON and CSV export; test suite.
- Secrets are read only from the environment (`.env`); no credentials in the code.
