# SPARTAN — Structured Parsing and Relevant Table Analysis

[![tests](https://github.com/shlok-n/spartan/actions/workflows/tests.yml/badge.svg)](https://github.com/shlok-n/spartan/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/shlok-n/spartan/blob/main/LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)
[![Paper](https://img.shields.io/badge/paper-Scientific%20Reports-informational.svg)](https://doi.org/10.1038/s41598-026-44325-7)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](https://github.com/shlok-n/spartan/blob/main/CONTRIBUTING.md)

Heuristic, training-free table detection and extraction for PDF documents. No GPU,
no model training, fully explainable.

> **Open source and open to contributions.** Bug reports, difficult sample documents, new
> OCR engines and better heuristics are all welcome — see [Contributing](#contributing).

Reference implementation of:

> Nandurbarkar, S., Chaudhari, A. Y. & Mulla, R. *SPARTAN: automated table detection and
> extraction from documents using advanced OpenCV heuristics and OCR techniques.*
> Scientific Reports **16**, 14954 (2026). https://doi.org/10.1038/s41598-026-44325-7

---

## Highlights

- **Exact text for born-digital PDFs.** Characters come from the PDF text layer and are
  mapped onto the detected cell grid, so part numbers are reproduced exactly (no
  `0`/`O` or `1`/`I` OCR confusions). OCR (Tesseract, offline) is used only for scanned
  pages or for regions that have no text layer.
- **Bordered, borderless and fill-delimited tables**: morphological boundary detector,
  alignment-based borderless detector with pseudo boundaries, and adaptive thresholding
  for tables whose cells are delimited only by background colours.
- **Complex structure**: nested (multi-level) headers, row/column-spanning cells, title
  rows, section rows, key-value tables and form grids, header-less list tables, and tables
  drawn as several separate boxes.
- **Document-level post-processing**: tables continuing across pages are merged (repeated
  header rows removed), and repeated page headers/footers are detected and de-duplicated.
- **Optional PCN schema mapping** (the paper's "LLM hook"). Deterministic keyword
  matching first; a local LLM through Ollama (or Azure OpenAI) fills the remaining gaps.
- CLI, Python API and REST API (FastAPI); JSON and CSV output; debug visualisations.

## Installation

Python 3.10 or newer.

```bash
pip install spartan-tables                  # core: born-digital PDFs
pip install "spartan-tables[ocr]"           # + Tesseract OCR for scanned pages
pip install "spartan-tables[all]"           # + OCR, REST API, local LLM, memory stats
```

Extras: `ocr`, `api`, `llm`, `azure`, `metrics`, `all`. The package is imported as `spartan`.

From source (for development):

```bash
git clone https://github.com/shlok-n/spartan.git spartan && cd spartan
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[all,dev]"                           # or: pip install -r requirements.txt
```

Optional components:

| Component | Needed for | Install |
|---|---|---|
| Tesseract binary | scanned pages / `--text-source ocr` | Windows: `winget install UB-Mannheim.TesseractOCR` · macOS: `brew install tesseract` · Debian/Ubuntu: `apt install tesseract-ocr` |
| Ollama + a model | `--pcn` with LLM gap-filling | <https://ollama.com>, e.g. `ollama pull gemma4` |
| Azure AI Vision / Azure OpenAI | optional cloud OCR / LLM | your own Azure resources, `pip install -e ".[azure]"` |

Born-digital PDFs need neither OCR nor an LLM.

## Configuration and credentials

Runtime settings are read from environment variables prefixed with `SPARTAN_`, or from a
`.env` file in the working directory:

```bash
cp .env.example .env      # then edit .env and add your own values
```

**The repository contains no credentials.** Cloud services are optional; to use them, put
your own endpoint and key in `.env` (git-ignored). For example:

```ini
SPARTAN_AZURE_VISION_ENDPOINT=https://<your-resource>.cognitiveservices.azure.com/
SPARTAN_AZURE_VISION_KEY=<your-azure-vision-key>
```

If a selected service is not configured, SPARTAN logs a warning explaining which variables
to set and continues without it. See `.env.example` for every setting.

## OCR engines

OCR is only needed for pages without a text layer (scans) or with `--text-source ocr`.
Choose the engine with `--ocr-engine` (or `PipelineConfig(ocr_engine=...)`):

| Engine | Runs | Setup |
|---|---|---|
| `tesseract` (default) | offline | install the Tesseract binary; set `SPARTAN_TESSERACT_CMD` if it is not on `PATH` |
| `azure` | Azure AI Vision Read API | `pip install -e ".[azure]"`, set `SPARTAN_AZURE_VISION_ENDPOINT` and `SPARTAN_AZURE_VISION_KEY` |
| `none` | — | text layer only; scanned tables are detected but their cells stay empty |
| `package.module:Class` | anything | your own engine, see below |

`--ocr-scope` controls how OCR is applied: `page` (one call per page, words assigned to cells
geometrically; the default) or the paper's cell strategies `single`, `multi` (cells stacked in
chunks of `--chunk-size` with the row-separator image) and `auto`.

### Using another OCR engine

Any object with a `name`, `image_to_words(image)` and `image_to_text(image)` works (`image` is
a BGR NumPy array). [`examples/custom_ocr_engine.py`](https://github.com/shlok-n/spartan/blob/main/examples/custom_ocr_engine.py) wraps
EasyOCR in about 40 lines. Use it without modifying the package:

```bash
python -m spartan --file scan.pdf --ocr-engine examples.custom_ocr_engine:EasyOcrEngine
```

or register it under a short name from Python:

```python
from spartan import PipelineConfig, process_pdf
from spartan.ocr import register_ocr_engine
from examples.custom_ocr_engine import EasyOcrEngine

register_ocr_engine("easyocr", EasyOcrEngine)
document = process_pdf("scan.pdf", PipelineConfig(ocr_engine="easyocr"))
```

## Quick start

```bash
# Standard profile (pre-processing + region segmentation)
python -m spartan --file notice.pdf

# Every PDF in a folder, with CSV export and debug images
python -m spartan --file ./pdfs --csv --save

# Multi-column documents (e.g. scientific papers)
python -m spartan --file paper.pdf --preprocess --column --region

# Map to the PCN schema (Ollama by default, deterministic only with --llm-provider none)
python -m spartan --file notice.pdf --pcn
```

Results are written to `output/<document>/<document>.json` (plus `csv/table_<n>.csv` and
`<document>_pcn.json` when requested).

## Using SPARTAN in your code

```python
import spartan

document = spartan.process_pdf("notice.pdf")
for table in document["tables"]:
    print(table["title"], table["columns"], len(table["rows"]))
```

Tables load directly into pandas:

```python
import pandas as pd

frames = [pd.DataFrame(t["rows"], columns=t["columns"]) for t in document["tables"]]
```

Options are set with `PipelineConfig` (all fields have sensible defaults):

```python
from spartan import PipelineConfig, SpartanPipeline, TextSourceMode

config = PipelineConfig(
    column_detection=True,               # multi-column layouts
    text_source=TextSourceMode.AUTO,     # PDF text layer, OCR only where needed
    ocr_engine="tesseract",              # or "azure", "none", "package.module:Class"
    dpi=300,
)
pipeline = SpartanPipeline(config)       # reuse one pipeline for many files
for path in ["a.pdf", "b.pdf"]:
    document = pipeline.process(path)
```

`process_pdf` returns a JSON-serialisable `dict` (format below). `SpartanPipeline.process`
also includes the internal `Table` objects under `document["_tables"]`, and
`SpartanPipeline.process_and_save` writes JSON/CSV files like the CLI does.

## CLI reference (paper Table 2)

| Flag | Stage | Description |
|---|---|---|
| `--preprocess` | Pre-processing | grayscale, adaptive thresholding, fixed-DPI normalisation |
| `--column` | Column detection | split multi-column layouts at vertical whitespace gutters |
| `--region` | Region segmentation | table detection; pre-/inter-/post-text segmentation |
| *(implicit)* | Table parsing | LSD line detection, grid and spanning-cell recovery |
| `--text-source` | Data curation | `auto` (text layer, OCR fallback), `pdf`, `ocr` |
| `--ocr-engine` | Data curation | `tesseract` (default), `azure`, `none` |
| `--ocr-scope` / `--chunk-size` | Data curation | `page` (OCR once, assign words) or the paper's `single` / `multi` / `auto` cell strategies |
| *(implicit)* | Post-processing | cross-page merging, header/footer de-duplication |
| `--pcn`, `--llm-provider`, `--llm-model` | LLM hook | optional PCN schema mapping |
| `--save`, `--csv`, `--stdout`, `--edge-detection`, `--dpi`, `--output` | Output | intermediate artefacts, CSV, JSON to stdout, Canny map, resolution, folder |

As in the paper, a step runs only when its flag is given. Without any step flag, the
standard profile `--preprocess --region` (the evaluation configuration) is used.

## Output format

```jsonc
{
  "document_name": "Analog Devices",
  "tables": [{
    "id": "table_2",
    "pages": [1, 2],                      // merged across the page break
    "type": "normal",                     // normal | nested | key_value
    "title": "PDN 23_0061 Discontinued Parts Material List",
    "title_source": "table",              // table | context | none
    "columns": ["Model", "Product Family", "Replacement Part", "Pin To Pin Compatible", "Comments"],
    "header_rows": 1,
    "rows": [["ADAU1328BSTZ", "ADAU1328", "AD1938WBSTZ", "Yes", "None"], ...],
    "merged_cells": [{"row": 4, "col": 2, "row_span": 2, "col_span": 1, "merge_type": "row", "text": "..."}],
    "pairs": [...],                       // key_value tables only
    "column_widths": [...], "detection_method": "boundary", "text_source": "pdf_text_layer",
    "sources": [{"page": 1, "segment": "table_2", "bbox": [...]}, ...]
  }],
  "text": [{"page": 1, "id": "pre_text", "kind": "pre_text", "text": "...", "boilerplate": false}],
  "full_text": "...",
  "pages": [...],                         // per-page segments in reading order, timings
  "processing": {"pages": 2, "seconds": 2.4, "seconds_per_page": 1.2, "peak_rss_mb": 185.0},
  "pcn": {...}                            // only with --pcn
}
```

Merged (spanning) cells are expanded: their value is repeated in every row and column
they cover, as described in the paper, and listed in `merged_cells`.

## REST API

```bash
uvicorn spartan.api:app --host 127.0.0.1 --port 8000     # or: python -m spartan.api
```

| Method | Path | Returns |
|---|---|---|
| `GET` | `/health` | status and version |
| `POST` | `/extract?column_detection=false&pcn=false` | full document JSON |
| `POST` | `/upload-pdf` | PCN schema (used by the SPARTAN front-end) |

Uploads must be PDFs (checked by content) and are limited to `SPARTAN_MAX_UPLOAD_MB`.
Errors are returned as `{"error": "..."}`.

## PCN schema mapping

`spartan/assets/fixed_schema.json` defines the output fields and
`spartan/assets/configuration.json` lists example labels for each field. Mapping order:

1. Table columns are matched to the table fields (`Impacted_Part`, `Replacement_Part`,
   `Pin_to_Pin_Match`). Part fields also require the column content to look like part
   numbers. Repeated column groups (`EOL Devices | Replacements | EOL Devices | …`) are
   unrolled.
2. Tables without matching columns, but with a title such as "Affected Part Numbers",
   contribute all their cells to `Impacted_Part`.
3. Scalar fields come from key-value tables, matching table columns and `Label: value`
   lines. Dates must be parseable and are normalised to `YYYY-MM-DD`.
4. Only what is still missing is sent to the LLM, with JSON-schema constrained output at
   temperature 0.

How each value was obtained is recorded in `document["pcn_provenance"]`.

## Architecture

| Module | Paper section |
|---|---|
| `pre_processing.py` | Pre-processing |
| `column_detection.py` | Column detection |
| `region_segmentation.py` | Region segmentation: crop, boundary detector, text detector, unification, segmentation |
| `table_parser.py` | Table parsing (LSD, line merging, grid / spanning cells, refinement) |
| `data_curation.py` | Data curation (table types, nested headers, merged cells, OCR correction, per-page merge) |
| `text_extraction.py`, `ocr.py` | Text layer, OCR engines, single/multi-cell OCR, row-separator chunking |
| `post_processing.py`, `table_merging.py` | Post-processing (cross-page merge, boilerplate removal) |
| `plugins/pcn.py`, `plugins/llm.py` | LLM hook |
| `config.py` | All heuristic thresholds (defined at 300 DPI and scaled automatically) and environment settings |
| `pipeline.py`, `cli.py`, `api.py`, `export.py`, `visualization.py` | Orchestration and interfaces |

## Testing

```bash
python -m pytest
```

Unit tests use synthetic images and need no external services. End-to-end tests run on real
PCN/PDN PDFs from `../Dataset` (or the folder in `SPARTAN_DATASET_DIR`) and are skipped when
it is absent; the documents are third-party and are not distributed with this repository.

## Performance (23 PCN/PDN PDFs, ~90 pages, CPU only)

About 0.3–1.5 s per page (the first pages include start-up), and about 280 MB peak memory,
without OCR. Measured on the development laptop. The LLM step adds roughly a minute per
document with an 8B model on CPU.

## Known limitations

- Scanned pages need the Tesseract binary; without it, tables on such pages are detected
  but their cells stay empty (a warning is logged).
- Heuristics assume roughly upright pages; heavily skewed scans are not de-skewed.
- Word splitting relies on glyph geometry; PDFs that break a text run in the middle of a
  word can still produce a stray split (e.g. "Withdrawa l").
- Small borderless blocks whose first row is a set of `Label:` headers can be misread as
  key-value tables.
- The borderless detector needs at least three aligned rows and two columns.

## Contributing

Contributions of any size are welcome, and you don't need permission to start:

- **Found a table SPARTAN gets wrong?** [Open an issue](https://github.com/shlok-n/spartan/issues/new/choose) with the
  document or a screenshot. Hard examples are the most valuable contribution.
- **Want to code?** Fork the repository, make your change with a test, and open a pull
  request. [CONTRIBUTING.md](https://github.com/shlok-n/spartan/blob/main/CONTRIBUTING.md) explains the setup and lists open tasks, from
  small fixes to de-skewing and benchmark evaluation.
- **Using SPARTAN somewhere?** Let us know in an issue or discussion; it helps prioritise.

Please follow the [Code of Conduct](https://github.com/shlok-n/spartan/blob/main/CODE_OF_CONDUCT.md). Report security issues privately
as described in [SECURITY.md](https://github.com/shlok-n/spartan/blob/main/SECURITY.md).

## License

Released under the [MIT License](https://github.com/shlok-n/spartan/blob/main/LICENSE).

## Citation

If you use SPARTAN in your work, please cite the paper (see also [`CITATION.cff`](https://github.com/shlok-n/spartan/blob/main/CITATION.cff)):

```bibtex
@article{nandurbarkar2026spartan,
  title   = {{SPARTAN}: automated table detection and extraction from documents using
             advanced {OpenCV} heuristics and {OCR} techniques},
  author  = {Nandurbarkar, S. and Chaudhari, A. Y. and Mulla, R.},
  journal = {Scientific Reports},
  volume  = {16},
  pages   = {14954},
  year    = {2026},
  doi     = {10.1038/s41598-026-44325-7}
}
```
