# Changelog

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
