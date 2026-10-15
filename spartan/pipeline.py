"""Pipeline orchestration: PDF in, structured document out."""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium

from . import __version__
from . import visualization as viz
from .column_detection import detect_columns
from .config import PipelineConfig
from .data_curation import curate_segments
from .export import public_document, write_csvs, write_json
from .models import BBox, ImageRegion, PageImage, PageResult, Segment, SegmentKind, Word, words_in_box
from .ocr import OcrEngine, create_ocr_engine
from .post_processing import build_document
from .pre_processing import build_page_image, save_page_images
from .region_segmentation import detect_tables, segment_region
from .text_extraction import acquire_page_words

log = logging.getLogger(__name__)


def _peak_rss_mb() -> float | None:
    try:
        import psutil
    except ImportError:
        return None
    info = psutil.Process().memory_info()
    peak = getattr(info, "peak_wset", None) or getattr(info, "rss", 0)
    return round(peak / (1024 * 1024), 1)


class SpartanPipeline:
    """Runs the full SPARTAN pipeline on PDF files.

    >>> pipeline = SpartanPipeline(PipelineConfig())
    >>> document = pipeline.process("notice.pdf")
    >>> document["tables"][0]["columns"]
    """

    def __init__(self, config: PipelineConfig | None = None, ocr_engine: OcrEngine | None = None):
        self.config = config or PipelineConfig()
        self.ocr_engine = ocr_engine if ocr_engine is not None else create_ocr_engine(
            self.config.ocr_engine, self.config.ocr_params
        )

    # ------------------------------------------------------------------ #
    def process(self, pdf_path: str | Path, document_name: str | None = None) -> dict[str, Any]:
        """Process one PDF and return the document dictionary."""
        cfg = self.config
        pdf_path = Path(pdf_path)
        name = document_name or pdf_path.stem
        out_dir = cfg.output_dir / name
        if cfg.save_intermediate and out_dir.exists():
            shutil.rmtree(out_dir)

        log.info("Processing %s (column=%s, region=%s, preprocess=%s, text=%s)", pdf_path.name,
                 cfg.column_detection, cfg.region_segmentation, cfg.preprocess, cfg.text_source.value)
        start = time.perf_counter()
        pages: list[PageResult] = []
        pdf = pdfium.PdfDocument(str(pdf_path))
        try:
            for index in range(len(pdf)):
                page = pdf[index]
                try:
                    pages.append(self._process_page(page, index + 1, out_dir))
                finally:
                    page.close()
        finally:
            pdf.close()

        document = build_document(name, pages, cfg)
        elapsed = time.perf_counter() - start
        document["source_file"] = pdf_path.name
        document["spartan_version"] = __version__
        document["settings"] = {
            "dpi": cfg.dpi,
            "preprocess": cfg.preprocess,
            "column_detection": cfg.column_detection,
            "region_segmentation": cfg.region_segmentation,
            "text_source": cfg.text_source.value,
            "ocr_engine": self.ocr_engine.name if self.ocr_engine else "none",
            "ocr_scope": cfg.ocr_params.scope.value,
        }
        document["processing"] = {
            "pages": len(pages),
            "seconds": round(elapsed, 2),
            "seconds_per_page": round(elapsed / max(1, len(pages)), 2),
            "peak_rss_mb": _peak_rss_mb(),
        }

        if cfg.pcn_mapping:
            from .plugins.pcn import map_to_pcn_schema

            document["pcn"] = map_to_pcn_schema(document)

        log.info("Finished %s: %d page(s), %d table(s) in %.1fs", pdf_path.name, len(pages),
                 len(document["tables"]), elapsed)
        return document

    def process_and_save(self, pdf_path: str | Path) -> tuple[dict[str, Any], Path]:
        """Process a PDF and write ``<output>/<name>/<name>.json`` (+ CSVs if enabled)."""
        document = self.process(pdf_path)
        folder = self.config.output_dir / document["document_name"]
        json_path = write_json(document, folder / f"{document['document_name']}.json")
        if self.config.export_csv:
            write_csvs(document["_tables"], folder / "csv")
        if "pcn" in document:
            write_json(document["pcn"], folder / f"{document['document_name']}_pcn.json")
        return document, json_path

    # ------------------------------------------------------------------ #
    def _process_page(self, page: pdfium.PdfPage, page_number: int, out_dir: Path) -> PageResult:
        cfg = self.config
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        page_image = build_page_image(page, page_number, cfg)
        timings["preprocess"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        words, text_source = acquire_page_words(page, page_image, cfg, self.ocr_engine)
        timings["text"] = time.perf_counter() - t0

        page_dir = out_dir / f"page_{page_number}"
        if cfg.save_intermediate:
            save_page_images(page_image, page_dir, cfg)

        # Column detection (optional)
        t0 = time.perf_counter()
        regions = self._regions(page_image, words, page_dir)
        timings["columns"] = time.perf_counter() - t0

        result = PageResult(page_number, page_image.width, page_image.height, text_source,
                            regions=[r.bbox for r in regions])
        t_region = t_curate = 0.0
        multi = len(regions) > 1
        for region in regions:
            t0 = time.perf_counter()
            if cfg.region_segmentation:
                segments, crop, debug = segment_region(region.image, region.words, cfg)
                if cfg.save_intermediate:
                    viz.save(viz.draw_regions(region.image, crop, debug),
                             page_dir / f"region_{region.index}_tables.png")
            else:
                segments = [Segment("pre_text", SegmentKind.PRE_TEXT,
                                    BBox(0, 0, region.image.shape[1], region.image.shape[0]),
                                    region.image, region.words)]
            if multi:
                for seg in segments:
                    seg.id = f"col{region.index + 1}_{seg.id}"
            t_region += time.perf_counter() - t0

            t0 = time.perf_counter()
            debug_parse: dict | None = {} if cfg.save_intermediate else None
            items = curate_segments(segments, page_number, region.bbox, cfg, text_source,
                                    self.ocr_engine, debug_parse)
            result.items.extend(items)
            t_curate += time.perf_counter() - t0

            if cfg.save_intermediate:
                for seg in segments:
                    viz.save(seg.image, page_dir / f"{seg.id}.png")
                for seg, parsed, tables in (debug_parse or {}).get("parsed", []):
                    viz.save(viz.draw_parsed_table(seg.image, parsed, tables), page_dir / f"{seg.id}_grid.png")

        timings["regions"] = t_region
        timings["curation"] = t_curate
        result.timings = timings
        if cfg.save_intermediate:
            write_json({"page": result.to_dict()}, page_dir / f"page_{page_number}.json")
        log.info("Page %d: %d table(s), %d text block(s) [%s]", page_number, len(result.tables),
                 len(result.text_blocks), text_source.value)
        return result

    def _regions(self, page_image: PageImage, words: list[Word], page_dir: Path) -> list[ImageRegion]:
        cfg = self.config
        full = BBox(0, 0, page_image.width, page_image.height)
        if not cfg.column_detection:
            return [ImageRegion(0, full, page_image.bgr, list(words))]
        # Tables must not be split by column detection: find them on the full page first.
        crop, tables, _ = detect_tables(page_image.bgr, words, cfg)
        table_boxes = [t.bbox.translate(crop.x0, crop.y0) for t in tables]
        regions, gaps = detect_columns(page_image, words, cfg, table_boxes)
        if cfg.save_intermediate:
            viz.save(viz.draw_columns(page_image.bgr, gaps, [r.bbox for r in regions]),
                     page_dir / "columns.png")
        return regions or [ImageRegion(0, full, page_image.bgr, words_in_box(words, full))]


def process_pdf(pdf_path: str | Path, config: PipelineConfig | None = None) -> dict[str, Any]:
    """Convenience wrapper returning the JSON-serialisable document."""
    return public_document(SpartanPipeline(config).process(pdf_path))
