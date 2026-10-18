"""SPARTAN - Structured Parsing and Relevant Table Analysis.

Heuristic, training-free table detection and extraction for PDF documents.

    >>> import spartan
    >>> document = spartan.process_pdf("notice.pdf")
    >>> for table in document["tables"]:
    ...     print(table["title"], table["columns"], len(table["rows"]))

Nandurbarkar, Chaudhari & Mulla, "SPARTAN: automated table detection and
extraction from documents using advanced OpenCV heuristics and OCR techniques",
Scientific Reports 16:14954 (2026). https://doi.org/10.1038/s41598-026-44325-7
"""

__version__ = "2.2.0"

from .config import OcrScope, PipelineConfig, TextSourceMode  # noqa: E402

__all__ = [
    "PipelineConfig",
    "TextSourceMode",
    "OcrScope",
    "SpartanPipeline",
    "process_pdf",
    "register_ocr_engine",
    "__version__",
]

_LAZY = {
    "SpartanPipeline": "pipeline",
    "process_pdf": "pipeline",
    "register_ocr_engine": "ocr",
}


def __getattr__(name):  # lazy imports keep `import spartan` light
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(f".{_LAZY[name]}", __name__), name)
    raise AttributeError(f"module 'spartan' has no attribute {name!r}")
