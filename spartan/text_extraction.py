"""Text acquisition: PDF text layer first, OCR as fallback.

SPARTAN targets born-digital PDFs. These already contain the exact characters
and their positions, so OCR-ing a rendered image of them only adds errors
(``0``/``O``, ``1``/``I``, ...). This module therefore:

1. Reads every character and its box from the PDF text layer (pdfium) and groups
   them into words, mapped into the pixel space of the rendered page.
2. Decides per page whether OCR is needed: when the page has (almost) no text
   layer it is treated as scanned, and when a significant amount of ink is not
   explained by text or ruling lines (e.g. a table pasted as a picture) the page
   is processed in *hybrid* mode, where OCR words are added only where the text
   layer has nothing.
3. Assigns words to table cells geometrically.
"""

from __future__ import annotations

import ctypes
import logging
import unicodedata
from typing import Sequence

import cv2
import numpy as np
import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c

from .config import PipelineConfig, TextSourceMode
from .models import BBox, GridCell, PageImage, TextSource, Word, words_to_text
from .ocr import OcrEngine

log = logging.getLogger(__name__)

_WHITESPACE = {" ", "\t", "\r", "\n", "Â ", "â€‚", "â€ƒ", "â€‰", "ï¿¾", "\x00"}
_WORD_GAP_RATIO = 0.12     # minimum word gap, as a fraction of the line height
_NOISE_FACTOR = 2.8        # word gap must exceed this multiple of the line's median glyph gap
_LOOSE_GAP_RATIO = 0.05    # ...and the advance boxes must be separated by this fraction of the height


def _is_private_use(ch: str) -> bool:
    """Symbol-font glyphs (bullets, arrows) are mapped to the Private Use Area."""
    return "î€€" <= ch <= "ï£¿"


# --------------------------------------------------------------------------- #
# PDF text layer
# --------------------------------------------------------------------------- #
def _page_to_device_transform(page: pdfium.PdfPage, width: int, height: int) -> np.ndarray:
    """Affine transform (2x3) mapping PDF page coordinates to rendered pixels.

    Uses pdfium's own ``FPDF_PageToDevice`` so page rotation and crop boxes are
    handled exactly as during rendering.
    """
    dx, dy = ctypes.c_int(), ctypes.c_int()

    def to_device(x: float, y: float) -> tuple[float, float]:
        # Scale up by 100 to recover sub-pixel precision from the integer API.
        pdfium_c.FPDF_PageToDevice(page.raw, 0, 0, width * 100, height * 100, 0, x, y, dx, dy)
        return dx.value / 100.0, dy.value / 100.0

    left, bottom, right, top = page.get_cropbox()
    src = np.float32([[left, bottom], [right, bottom], [left, top]])
    dst = np.float32([to_device(*p) for p in src])
    return cv2.getAffineTransform(src, dst)


def extract_pdf_words(page: pdfium.PdfPage, image_width: int, image_height: int) -> tuple[list[Word], int]:
    """Read words from the PDF text layer. Returns ``(words, number_of_chars)``."""
    textpage = page.get_textpage()
    try:
        n_chars = textpage.count_chars()
        if n_chars <= 0:
            return [], 0
        transform = _page_to_device_transform(page, image_width, image_height)

        glyphs: list[_Glyph | None] = []          # None marks a space / unusable char
        for i in range(n_chars):
            code = pdfium_c.FPDFText_GetUnicode(textpage.raw, i)
            ch = chr(code) if code else "\x00"
            if pdfium_c.FPDFText_IsGenerated(textpage.raw, i) == 1 or ch in _WHITESPACE or _is_private_use(ch):
                glyphs.append(None)
                continue
            try:
                loose = textpage.get_charbox(i, loose=True)
                tight = textpage.get_charbox(i, loose=False)
            except pdfium.PdfiumError:
                glyphs.append(None)
                continue
            box, tbox = _to_pixels(loose, transform), _to_pixels(tight, transform)
            glyphs.append(_Glyph(ch, box, tbox) if box.width > 0 and box.height > 0 else None)
    finally:
        textpage.close()

    return _glyphs_to_words(glyphs), n_chars


class _Glyph:
    __slots__ = ("ch", "box", "tight")

    def __init__(self, ch: str, box: BBox, tight: BBox):
        self.ch, self.box, self.tight = ch, box, tight


def _to_pixels(rect: tuple[float, float, float, float], transform: np.ndarray) -> BBox:
    left, bottom, right, top = rect
    pts = np.array([[left, bottom, 1.0], [right, top, 1.0]]) @ transform.T
    return BBox(float(min(pts[:, 0])), float(min(pts[:, 1])), float(max(pts[:, 0])), float(max(pts[:, 1])))


def _glyphs_to_words(glyphs: list["_Glyph | None"]) -> list[Word]:
    """Group glyphs into words: split on spaces, line changes and word-sized gaps.

    Some PDFs position words without emitting space characters, others use
    letter-spacing so glyphs inside a word are not contiguous. A fixed gap
    threshold cannot serve both, so the threshold adapts to each text line: a
    gap between *tight* glyph boxes starts a new word when it exceeds
    ``_NOISE_FACTOR`` times the line's median glyph gap (and at least
    ``_WORD_GAP_RATIO`` times the line height).
    """
    lines: list[list[_Glyph | None]] = [[]]
    last: _Glyph | None = None
    for g in glyphs:
        if g is not None and last is not None:
            height = max(min(last.box.height, g.box.height), 1.0)
            if abs(g.box.cy - last.box.cy) > 0.5 * height or g.box.x0 - last.box.x1 < -1.5 * height:
                lines.append([])
        lines[-1].append(g)
        if g is not None:
            last = g
    words: list[Word] = []
    for line in lines:
        words.extend(_line_to_words(line))
    return words


def _line_to_words(line: list["_Glyph | None"]) -> list[Word]:
    gaps, heights = [], []
    prev: _Glyph | None = None
    for g in line:
        if g is None:
            prev = None
            continue
        heights.append(g.box.height)
        if prev is not None:
            gaps.append(max(0.0, g.tight.x0 - prev.tight.x1))
        prev = g
    if not heights:
        return []
    height = float(np.median(heights))
    noise = float(np.median(gaps)) if gaps else 0.0
    threshold = max(_WORD_GAP_RATIO * height, _NOISE_FACTOR * noise)

    words: list[Word] = []
    current: list[_Glyph] = []

    def flush() -> None:
        if not current:
            return
        raw_text = "".join(g.ch for g in current)
        try:
            raw_text = raw_text.encode("utf-16", "surrogatepass").decode("utf-16")
        except UnicodeError:
            pass
        text = unicodedata.normalize("NFKC", raw_text).strip()
        if text:
            words.append(Word(
                text=text,
                bbox=BBox.union_all(g.box for g in current),
                source="pdf",
                ink_bbox=BBox.union_all(g.tight for g in current),
            ))
        current.clear()

    for g in line:
        if g is None:
            flush()
            continue
        if current:
            prev_g = current[-1]
            # Narrow glyphs (1, l, i) have large tight gaps but touching advance boxes,
            # so the loose boxes must be separated as well.
            if g.tight.x0 - prev_g.tight.x1 > threshold and g.box.x0 - prev_g.box.x1 > _LOOSE_GAP_RATIO * height:
                flush()
        current.append(g)
    flush()
    return words




# --------------------------------------------------------------------------- #
# OCR decision and hybrid merge
# --------------------------------------------------------------------------- #
def unexplained_ink_ratio(page: PageImage, words: Sequence[Word], config: PipelineConfig) -> float:
    """Fraction of the page covered by ink that is neither text-layer text nor ruling lines."""
    ink = (page.gray < config.preprocess_params.ink_threshold).astype(np.uint8) * 255
    pad = config.px(4)
    for w in words:
        b = w.bbox.expand(pad).clip(page.width, page.height).to_int()
        ink[int(b.y0):int(b.y1), int(b.x0):int(b.x1)] = 0
    k = config.px(40)
    lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))
    lines |= cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, k)))
    ink[lines > 0] = 0
    # Ignore isolated specks.
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return float(np.count_nonzero(ink)) / float(ink.size)


def _merge_ocr_words(pdf_words: list[Word], ocr_words: list[Word]) -> list[Word]:
    """Keep all PDF words and add OCR words only where the text layer has nothing."""
    merged = list(pdf_words)
    for ow in ocr_words:
        if not any(ow.bbox.overlap_ratio(pw.bbox) > 0.3 for pw in pdf_words):
            merged.append(ow)
    return merged


def acquire_page_words(
    page: pdfium.PdfPage,
    page_image: PageImage,
    config: PipelineConfig,
    ocr_engine: OcrEngine | None,
) -> tuple[list[Word], TextSource]:
    """Return all words of a page (pixel coordinates) and where they came from."""
    mode = config.text_source
    pdf_words: list[Word] = []
    n_chars = 0
    if mode is not TextSourceMode.OCR:
        pdf_words, n_chars = extract_pdf_words(page, page_image.width, page_image.height)

    if mode is TextSourceMode.PDF:
        return pdf_words, TextSource.PDF if pdf_words else TextSource.NONE

    if mode is TextSourceMode.OCR:
        if ocr_engine is None:
            log.warning("Page %d: OCR requested but no OCR engine is available", page_image.page_number)
            return [], TextSource.NONE
        return ocr_engine.image_to_words(page_image.bgr), TextSource.OCR

    # AUTO
    if n_chars < config.ocr_params.min_text_layer_chars:
        if ocr_engine is None:
            log.warning(
                "Page %d has no usable text layer and no OCR engine is available; "
                "tables will be detected but cells will be empty", page_image.page_number,
            )
            return pdf_words, TextSource.PDF if pdf_words else TextSource.NONE
        log.info("Page %d: no text layer, running OCR (%s)", page_image.page_number, ocr_engine.name)
        return ocr_engine.image_to_words(page_image.bgr), TextSource.OCR

    if ocr_engine is not None:
        ratio = unexplained_ink_ratio(page_image, pdf_words, config)
        if ratio > config.ocr_params.unexplained_ink_ratio:
            log.info(
                "Page %d: %.1f%% of the page is ink without text layer, running hybrid OCR",
                page_image.page_number, 100 * ratio,
            )
            ocr_words = ocr_engine.image_to_words(page_image.bgr)
            return _merge_ocr_words(pdf_words, ocr_words), TextSource.HYBRID
    return pdf_words, TextSource.PDF


# --------------------------------------------------------------------------- #
# Word -> cell assignment
# --------------------------------------------------------------------------- #
def assign_words_to_cells(cells: Sequence[GridCell], words: Sequence[Word]) -> dict[int, str]:
    """Map each cell id to the text of the words that fall inside it.

    A word belongs to the cell containing its centre; words whose centre lies
    exactly on a boundary go to the cell they overlap most.
    """
    buckets: dict[int, list[Word]] = {c.id: [] for c in cells}
    for word in words:
        owner = next((c for c in cells if c.bbox.contains_point(word.bbox.cx, word.bbox.cy)), None)
        if owner is None:
            best, best_area = None, 0.0
            for c in cells:
                inter = c.bbox.intersection(word.bbox)
                if inter is not None and inter.area > best_area:
                    best, best_area = c, inter.area
            owner = best
        if owner is not None:
            buckets[owner.id].append(word)
    return {cid: words_to_text(ws, line_separator=" ") for cid, ws in buckets.items()}
