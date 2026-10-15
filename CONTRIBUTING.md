# Contributing to SPARTAN

Thanks for your interest! SPARTAN is open to contributions of every size: bug reports, sample
documents that break the pipeline, documentation fixes, new OCR engines, better heuristics and
evaluation work. You don't need to ask before starting; for larger changes, opening an issue
first helps avoid duplicated effort.

## Ways to contribute

- **Report a bug.** Open an issue with the PDF (or a page of it) and what you expected. A
  document that reproduces the problem is the most useful thing you can send.
- **Share hard documents.** Tables that SPARTAN gets wrong are how the heuristics improve.
  Only share documents you are allowed to share publicly.
- **Pick an open task.** See [Open tasks](#open-tasks) below or issues labelled
  `good first issue` / `help wanted`.
- **Improve the docs.** Typos, unclear explanations and missing examples all count.

## Development setup

```bash
git clone https://github.com/shlok-n/spartan.git spartan && cd spartan
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[all,dev]"
python -m pytest
```

Tesseract and Ollama are only needed if you work on OCR or the PCN mapping. End-to-end tests
on real PDFs run when `SPARTAN_DATASET_DIR` points to a folder of documents (see the README);
otherwise they are skipped.

## Making a change

1. Fork the repository and create a branch: `git checkout -b fix/short-description`.
2. Make the change. Keep it focused; unrelated clean-ups belong in a separate pull request.
3. Add or update tests in `tests/`. Heuristic changes should come with a synthetic test
   (see `tests/conftest.py` for helpers that draw tables) or a dataset test.
4. Run `python -m pytest` and make sure everything passes.
5. Open a pull request describing what changed, why, and how you tested it. Before/after
   output for an affected document is very helpful.

## Code guidelines

- Follow the style of the surrounding code: type hints, small functions, docstrings that say
  *why* a heuristic exists.
- Pixel thresholds go in the `*Params` dataclasses in `spartan/config.py`, expressed at
  300 DPI (they are scaled automatically). Avoid magic numbers in the modules.
- Never commit credentials, `.env` files or confidential documents.
- New optional dependencies go into an extra in `pyproject.toml`, not the core dependencies.

## Adding an OCR engine

Implement `name`, `image_to_words(image)` and `image_to_text(image)`, raise
`OcrUnavailableError` when the engine cannot run, and register it with
`register_ocr_engine`. `examples/custom_ocr_engine.py` is a complete template. Engines with
heavy dependencies are welcome as examples or as an optional extra.

## Open tasks

Good starting points, roughly from easiest to hardest:

- **Word splits across text runs.** Some PDFs break a run mid-word ("Withdrawa l"). Tune the
  gap threshold in `spartan/text_extraction.py` without merging tightly spaced table cells.
- **Header rows misread as key-value tables.** Small borderless blocks whose first row is
  all `Label:` cells should be treated as a header (`spartan/data_curation.py`).
- **Wrapped lines in borderless tables.** Join a line into the previous row when its first
  column is empty and the text continues the line above.
- **De-skewing scanned pages** before table detection (`spartan/pre_processing.py`).
- **More OCR engines** (EasyOCR, PaddleOCR, docTR) as optional extras.
- **Evaluation on public benchmarks** (ICDAR 2013, PubTables-1M, FinTabNet) with standard
  structure metrics (GriTS, TEDS), plus a script to reproduce the numbers.

## Releasing (maintainers)

1. Update `__version__` in `spartan/__init__.py` (the only place the version is defined;
   follow [semantic versioning](https://semver.org)) and add an entry to `CHANGELOG.md`.
2. Check the package locally: `python -m build && twine check --strict dist/*`.
3. Commit, then create a GitHub release with a tag like `v2.1.0`. The `release` workflow
   builds the package and publishes it to PyPI automatically.

## Questions

Open an issue or a discussion. Be kind and constructive; see the
[Code of Conduct](CODE_OF_CONDUCT.md).
