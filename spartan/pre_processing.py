"""Stage 1 - Pre-processing.

Renders every PDF page to an image at a fixed DPI and derives the images that the
later stages reuse:

* **Normalisation** - pages are rendered at a fixed resolution (300 DPI by
  default), so one pixel always corresponds to the same physical length no
  matter what page size or source the PDF came from. All heuristic thresholds
  are defined relative to this resolution.
* **Grayscale conversion** - removes redundant colour information.
* **Adaptive thresholding** - produces a binary ink mask that is robust to
  shading, coloured cell fills and uneven backgrounds. Tables whose cells are
  delimited only by fill colours (no ruling lines) become detectable because
  the fill edges turn into strokes.
* **Optional Canny edge map** - kept for inspection (``--edge-detection``).

Pages are rendered lazily, one at a time, to keep peak memory low.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import pypdfium2 as pdfium

from .config import PipelineConfig
from .models import PageImage

log = logging.getLogger(__name__)


def render_page(page: pdfium.PdfPage, dpi: int) -> np.ndarray:
    """Render a pdfium page to a BGR ``uint8`` image."""
    bitmap = page.render(scale=dpi / 72.0, may_draw_forms=True)
    try:
        array = bitmap.to_numpy()
        mode = bitmap.mode
    finally:
        bitmap.close()
    if array.ndim == 2:
        return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    if mode in ("BGRA", "BGRX"):
        return cv2.cvtColor(array, cv2.COLOR_BGRA2BGR)
    if mode in ("RGBA", "RGBX"):
        return cv2.cvtColor(array, cv2.COLOR_RGBA2BGR)
    if mode == "RGB":
        return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    return np.ascontiguousarray(array[:, :, :3])


def preprocess_image(bgr: np.ndarray, config: PipelineConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(gray, binary)`` for a BGR page image.

    ``binary`` is an ink mask (255 = ink). With pre-processing enabled it is
    produced by Gaussian adaptive thresholding as described in the paper;
    otherwise a plain global threshold is used.
    """
    params = config.preprocess_params
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if config.preprocess:
        block = params.adaptive_block_size | 1  # must be odd
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, params.adaptive_c
        )
    else:
        _, binary = cv2.threshold(gray, params.ink_threshold, 255, cv2.THRESH_BINARY_INV)
    return gray, binary


def edge_map(gray: np.ndarray, config: PipelineConfig) -> np.ndarray:
    """Canny edge overlay (optional, for inspection only)."""
    p = config.preprocess_params
    return cv2.Canny(gray, p.canny_low, p.canny_high)


def build_page_image(page: pdfium.PdfPage, page_number: int, config: PipelineConfig) -> PageImage:
    bgr = render_page(page, config.dpi)
    gray, binary = preprocess_image(bgr, config)
    width_pt, height_pt = page.get_size()
    return PageImage(
        page_number=page_number,
        bgr=bgr,
        gray=gray,
        binary=binary,
        dpi=config.dpi,
        width_pt=width_pt,
        height_pt=height_pt,
    )


def iter_page_images(pdf_path: str | Path, config: PipelineConfig) -> Iterator[PageImage]:
    """Yield pre-processed pages one by one (stand-alone use of this stage)."""
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        for index in range(len(pdf)):
            page = pdf[index]
            try:
                yield build_page_image(page, index + 1, config)
            finally:
                page.close()
    finally:
        pdf.close()


def save_page_images(page: PageImage, folder: Path, config: PipelineConfig) -> None:
    """Persist the pre-processing outputs of one page (``--save``)."""
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"page_{page.page_number}"
    cv2.imwrite(str(folder / f"{stem}_render.png"), page.bgr)
    cv2.imwrite(str(folder / f"{stem}_binary.png"), 255 - page.binary)
    if config.edge_detection:
        cv2.imwrite(str(folder / f"{stem}_edges.png"), edge_map(page.gray, config))
