"""Cell-level OCR strategies with a fake engine (no Tesseract needed)."""

from dataclasses import replace

import numpy as np

from spartan.config import OcrParams, OcrScope
from spartan.models import BBox, GridCell
from spartan.ocr import ocr_cells


class FakeEngine:
    name = "fake"

    def __init__(self, multi_answer: str):
        self.multi_answer = multi_answer
        self.calls: list[tuple[int, int]] = []

    def image_to_words(self, image):
        return []

    def image_to_text(self, image):
        self.calls.append(image.shape[:2])
        # Tall images are stacked chunks; small ones are single cells.
        return self.multi_answer if image.shape[0] > 100 else "single"


def _column(n: int) -> list[GridCell]:
    return [GridCell(i, i, 0, 1, 1, BBox(0, 40 * i, 200, 40 * (i + 1))) for i in range(n)]


def test_multi_cell_ocr_splits_on_row_separator():
    image = np.full((200, 200, 3), 255, np.uint8)
    engine = FakeEngine("AD1328 RSRSRSRS@ AD1401\nRS RSRS RS@ SSM2141")   # marker mangled by OCR
    params = replace(OcrParams(), scope=OcrScope.MULTI, chunk_size=3)
    texts = ocr_cells(image, _column(3), engine, params, shrink=3)
    assert [texts[i] for i in range(3)] == ["AD1328", "AD1401", "SSM2141"]
    assert len(engine.calls) == 1                     # one OCR call for the whole chunk


def test_multi_cell_sanity_check_falls_back_to_single():
    image = np.full((200, 200, 3), 255, np.uint8)
    engine = FakeEngine("only one row recognised")
    params = replace(OcrParams(), scope=OcrScope.MULTI, chunk_size=3)
    texts = ocr_cells(image, _column(3), engine, params, shrink=3)
    assert set(texts.values()) == {"single"}


def test_auto_scope_and_merged_cells():
    image = np.full((200, 400, 3), 255, np.uint8)
    cells = _column(2) + [GridCell(9, 0, 1, 2, 1, BBox(200, 0, 400, 80))]   # row-spanning cell
    engine = FakeEngine("x")
    params = replace(OcrParams(), scope=OcrScope.AUTO, auto_threshold=120)
    texts = ocr_cells(image, cells, engine, params, shrink=3)
    assert texts[9] == "single" and len(texts) == 3   # < 120 cells -> single-cell OCR
