"""Command-line interface (paper Table 2).

Examples::

    python -m spartan --file notice.pdf                       # default profile
    python -m spartan --file notice.pdf --preprocess --region --save
    python -m spartan --file paper.pdf --preprocess --column --region --csv
    python -m spartan --file ./pdfs --pcn --llm-provider ollama
    python -m spartan --file scan.pdf --text-source ocr --ocr-scope auto --chunk-size 15

Step flags follow the paper: a step runs only when its flag is given. When no
step flag is given at all, the standard profile (``--preprocess --region``, the
configuration used for the paper's evaluation) is used.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

from . import __version__
from .config import LlmProvider, OcrScope, PipelineConfig, TextSourceMode, get_settings
from .export import public_document

log = logging.getLogger("spartan")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="spartan",
        description="SPARTAN - heuristic table detection and extraction for PDF documents.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--file", "-f", required=True, type=Path, help="PDF file or folder of PDF files")
    p.add_argument("--output", "-o", type=Path, default=Path("output"), help="output folder")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    steps = p.add_argument_group("pipeline steps (paper Table 2)")
    steps.add_argument("--preprocess", action="store_true", help="grayscale + adaptive threshold pre-processing")
    steps.add_argument("--column", action="store_true", help="column detection for multi-column layouts")
    steps.add_argument("--region", action="store_true", help="region segmentation (table detection)")
    steps.add_argument("--edge-detection", action="store_true", help="also save a Canny edge map (with --save)")

    text = p.add_argument_group("text extraction / OCR")
    text.add_argument("--text-source", choices=[m.value for m in TextSourceMode], default="auto",
                      help="auto: PDF text layer, OCR only where needed")
    text.add_argument("--ocr-engine", default="tesseract", metavar="ENGINE",
                      help="tesseract (default), azure, none, or package.module:Class for a custom engine")
    text.add_argument("--ocr-scope", choices=[s.value for s in OcrScope], default="page",
                      help="page: OCR once and assign words; single/multi/auto: paper's cell OCR strategies")
    text.add_argument("--chunk-size", type=int, default=15, help="cells per chunk for multi-cell OCR")
    text.add_argument("--dpi", type=int, default=300, help="rendering resolution")

    out = p.add_argument_group("outputs")
    out.add_argument("--save", action="store_true", help="save intermediate images and per-page JSON")
    out.add_argument("--csv", action="store_true", help="also export every table as CSV")
    out.add_argument("--stdout", action="store_true", help="print the final JSON to stdout")

    llm = p.add_argument_group("optional PCN schema mapping (LLM hook)")
    llm.add_argument("--pcn", action="store_true", help="map the document to the PCN schema")
    llm.add_argument("--llm-provider", choices=[x.value for x in LlmProvider], default=None,
                     help="override SPARTAN_LLM_PROVIDER (none = deterministic mapping only)")
    llm.add_argument("--llm-model", default=None, help="override SPARTAN_OLLAMA_MODEL")

    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    any_step = args.preprocess or args.column or args.region
    if not any_step:
        log.info("No step flags given: using the standard profile (--preprocess --region)")
    base = PipelineConfig()
    return replace(
        base,
        dpi=args.dpi,
        preprocess=args.preprocess if any_step else True,
        column_detection=args.column,
        region_segmentation=args.region if any_step else True,
        edge_detection=args.edge_detection,
        text_source=TextSourceMode(args.text_source),
        ocr_engine=args.ocr_engine,
        save_intermediate=args.save,
        output_dir=args.output,
        export_csv=args.csv,
        pcn_mapping=args.pcn,
        ocr_params=replace(base.ocr_params, scope=OcrScope(args.ocr_scope), chunk_size=args.chunk_size),
    )


def _collect_pdfs(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.suffix.lower() == ".pdf")
    if path.is_file():
        return [path]
    raise FileNotFoundError(path)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")

    settings = get_settings()
    if args.llm_provider:
        settings.llm_provider = LlmProvider(args.llm_provider)
    if args.llm_model:
        settings.ollama_model = args.llm_model

    from .pipeline import SpartanPipeline  # heavy imports only when actually running

    try:
        pdfs = _collect_pdfs(args.file)
    except FileNotFoundError:
        log.error("Input not found: %s", args.file)
        return 2
    if not pdfs:
        log.error("No PDF files found in %s", args.file)
        return 2

    try:
        pipeline = SpartanPipeline(config_from_args(args))
    except ValueError as exc:            # e.g. unknown --ocr-engine
        log.error("%s", exc)
        return 2
    failures = 0
    for pdf in pdfs:
        try:
            document, json_path = pipeline.process_and_save(pdf)
            log.info("Wrote %s", json_path)
            if args.stdout:
                print(json.dumps(public_document(document), indent=2, ensure_ascii=False))
        except Exception:
            failures += 1
            log.exception("Failed to process %s", pdf)
    log.info("Done: %d processed, %d failed", len(pdfs) - failures, failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
