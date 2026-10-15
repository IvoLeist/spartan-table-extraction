"""Plugging your own OCR engine into SPARTAN.

An engine is any object with:

* ``name``                      - a short identifier (stored in the output metadata)
* ``image_to_words(image)``     - list of :class:`spartan.models.Word` with pixel boxes
* ``image_to_text(image)``      - plain text, lines separated by ``\\n``

``image`` is a BGR ``numpy`` array (OpenCV convention). Raise
:class:`spartan.ocr.OcrUnavailableError` from ``__init__`` if the engine cannot run;
SPARTAN then logs a warning and continues with the PDF text layer only.

Three ways to use the engine below (it needs ``pip install easyocr``):

1. Command line, without touching the package (run from the repository root)::

       python -m spartan --file scan.pdf --ocr-engine examples.custom_ocr_engine:EasyOcrEngine

2. Register it under a short name, then select it by that name::

       from spartan.ocr import register_ocr_engine
       register_ocr_engine("easyocr", EasyOcrEngine)
       process_pdf("scan.pdf", PipelineConfig(ocr_engine="easyocr"))

3. Pass an instance directly::

       SpartanPipeline(PipelineConfig(), ocr_engine=EasyOcrEngine(OcrParams(), get_settings()))
"""

from __future__ import annotations

import numpy as np

from spartan.config import OcrParams, Settings
from spartan.models import BBox, Word
from spartan.ocr import OcrUnavailableError


class EasyOcrEngine:
    name = "easyocr"

    def __init__(self, params: OcrParams, settings: Settings):
        try:
            import easyocr
        except ImportError as exc:
            raise OcrUnavailableError("pip install easyocr") from exc
        self._reader = easyocr.Reader(["en"], gpu=False)

    def image_to_words(self, image: np.ndarray) -> list[Word]:
        words = []
        for points, text, confidence in self._reader.readtext(image[:, :, ::-1]):   # BGR -> RGB
            xs, ys = [p[0] for p in points], [p[1] for p in points]
            # EasyOCR returns phrases; SPARTAN works with words, so split the box evenly.
            parts = text.split()
            if not parts:
                continue
            x0, x1, step = min(xs), max(xs), (max(xs) - min(xs)) / max(len(text), 1)
            cursor = x0
            for part in parts:
                width = step * len(part)
                words.append(Word(part, BBox(cursor, min(ys), min(cursor + width, x1), max(ys)),
                                  source="ocr", confidence=float(confidence)))
                cursor += width + step
        return words

    def image_to_text(self, image: np.ndarray) -> str:
        return "\n".join(self._reader.readtext(image[:, :, ::-1], detail=0, paragraph=False))
