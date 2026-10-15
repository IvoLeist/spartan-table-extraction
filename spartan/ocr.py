"""Pluggable OCR layer.

Engines implement :class:`OcrEngine`. Two are provided:

* :class:`TesseractEngine` - offline, the default (requires the Tesseract binary).
* :class:`AzureReadEngine` - Azure AI Vision Read API (optional; credentials come
  from ``SPARTAN_AZURE_VISION_ENDPOINT`` / ``SPARTAN_AZURE_VISION_KEY``).

Other engines (EasyOCR, PaddleOCR, a cloud service, ...) are added with
:func:`register_ocr_engine` or selected directly as ``package.module:Class``;
see ``examples/custom_ocr_engine.py``.

On top of the engines, this module implements the paper's cell-level OCR
strategies, used when ``--ocr-scope`` is ``single``, ``multi`` or ``auto``:

* **Single-cell OCR** - each cell is cropped (shrunk by 3 px on every side so
  borders are not read as ``|`` or ``!``) and recognised on its own.
* **Multi-cell OCR** - cells of a column are stacked in chunks of ``chunk_size``
  with the row-separator image between them; the recognised text is split on the
  separator marker. Merged cells are excluded from chunks and read individually.
  If the number of recovered rows does not match the chunk, the chunk falls back
  to single-cell OCR (sanity check from the paper).

It also contains the column-trend OCR auto-correction step.
"""

from __future__ import annotations

import importlib
import io
import logging
import re
import shutil
import time
from collections import Counter
from typing import Callable, Protocol, Sequence

import cv2
import numpy as np

from .config import OcrParams, OcrScope, Settings, get_settings
from .models import BBox, GridCell, Word

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Engine interface
# --------------------------------------------------------------------------- #
class OcrEngine(Protocol):
    name: str

    def image_to_words(self, image: np.ndarray) -> list[Word]:
        """Recognise words with pixel boxes (image coordinates)."""

    def image_to_text(self, image: np.ndarray) -> str:
        """Recognise plain text (lines separated by ``\\n``)."""


class OcrUnavailableError(RuntimeError):
    pass


class TesseractEngine:
    """Offline OCR with Tesseract (via ``pytesseract``)."""

    name = "tesseract"

    def __init__(self, params: OcrParams, settings: Settings | None = None):
        try:
            import pytesseract
        except ImportError as exc:
            raise OcrUnavailableError(
                "pytesseract is not installed: pip install \"spartan-tables[ocr]\" (plus the Tesseract binary)"
            ) from exc

        settings = settings or get_settings()
        if settings.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
        cmd = pytesseract.pytesseract.tesseract_cmd
        if shutil.which(cmd) is None and not (settings.tesseract_cmd and shutil.which(settings.tesseract_cmd)):
            raise OcrUnavailableError(
                "Tesseract binary not found. Install it (e.g. `winget install UB-Mannheim.TesseractOCR`) "
                "or set SPARTAN_TESSERACT_CMD."
            )
        self._tess = pytesseract
        self._params = params

    def image_to_words(self, image: np.ndarray) -> list[Word]:
        data = self._tess.image_to_data(
            image, config=self._params.page_tesseract_config, output_type=self._tess.Output.DICT
        )
        words = []
        for i, text in enumerate(data["text"]):
            text = (text or "").strip()
            conf = float(data["conf"][i])
            if not text or conf < 0:
                continue
            box = BBox.from_xywh(data["left"][i], data["top"][i], data["width"][i], data["height"][i])
            words.append(Word(text=text, bbox=box, source="ocr", confidence=conf / 100.0))
        return words

    def image_to_text(self, image: np.ndarray) -> str:
        text = self._tess.image_to_string(image, config=self._params.tesseract_config)
        return "\n".join(line.strip() for line in text.splitlines() if line.strip())


class AzureReadEngine:
    """Azure AI Vision Read API (cloud). Only used when explicitly selected."""

    name = "azure"

    def __init__(self, settings: Settings | None = None):
        settings = settings or get_settings()
        if not (settings.azure_vision_endpoint and settings.azure_vision_key):
            raise OcrUnavailableError(
                "Azure OCR selected but SPARTAN_AZURE_VISION_ENDPOINT / SPARTAN_AZURE_VISION_KEY are not set. "
                "Add your own endpoint and key to .env (see .env.example)."
            )
        try:
            from azure.cognitiveservices.vision.computervision import ComputerVisionClient
            from msrest.authentication import CognitiveServicesCredentials
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise OcrUnavailableError("Install the 'azure' extra: pip install spartan-tables[azure]") from exc
        self._client = ComputerVisionClient(
            settings.azure_vision_endpoint, CognitiveServicesCredentials(settings.azure_vision_key)
        )

    def _read(self, image: np.ndarray):
        ok, buf = cv2.imencode(".png", image)
        if not ok:
            raise ValueError("Failed to encode image for OCR")
        response = self._client.read_in_stream(io.BytesIO(buf.tobytes()), raw=True)
        operation_id = response.headers["Operation-Location"].rsplit("/", 1)[-1]
        delay = 0.25
        while True:
            result = self._client.get_read_result(operation_id)
            if str(result.status).lower() not in ("notstarted", "running", "operationstatuscodes.running",
                                                   "operationstatuscodes.not_started"):
                break
            time.sleep(delay)
            delay = min(delay * 2, 2.0)
        if str(result.status).lower().endswith("succeeded"):
            return result.analyze_result.read_results
        log.warning("Azure OCR failed with status %s", result.status)
        return []

    def image_to_words(self, image: np.ndarray) -> list[Word]:
        words = []
        for page in self._read(image):
            for line in page.lines:
                for w in line.words:
                    xs, ys = w.bounding_box[0::2], w.bounding_box[1::2]
                    words.append(Word(w.text, BBox(min(xs), min(ys), max(xs), max(ys)), "ocr", w.confidence))
        return words

    def image_to_text(self, image: np.ndarray) -> str:
        return "\n".join(line.text for page in self._read(image) for line in page.lines)


# --------------------------------------------------------------------------- #
# Engine registry
# --------------------------------------------------------------------------- #
EngineFactory = Callable[[OcrParams, Settings], OcrEngine]

_REGISTRY: dict[str, EngineFactory] = {
    "tesseract": lambda params, settings: TesseractEngine(params, settings),
    "azure": lambda params, settings: AzureReadEngine(settings),
}


def register_ocr_engine(name: str, factory: EngineFactory | None = None):
    """Register an OCR engine under ``name`` so it can be selected with ``--ocr-engine``.

    ``factory`` is called as ``factory(params, settings)`` and must return an object with
    ``name``, ``image_to_words(image)`` and ``image_to_text(image)`` (see :class:`OcrEngine`).
    A class with that constructor signature works directly. Usable as a decorator::

        @register_ocr_engine("easyocr")
        class EasyOcrEngine: ...
    """

    def _register(f: EngineFactory) -> EngineFactory:
        _REGISTRY[name.lower()] = f
        return f

    return _register(factory) if factory is not None else _register


def available_ocr_engines() -> list[str]:
    return sorted(_REGISTRY) + ["none"]


def _load_factory(spec: str) -> EngineFactory:
    """Resolve ``package.module:Attribute`` to an engine factory."""
    module_name, _, attr = spec.partition(":")
    try:
        return getattr(importlib.import_module(module_name), attr)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"Cannot load OCR engine {spec!r}: {exc}") from exc


def create_ocr_engine(name: str, params: OcrParams, settings: Settings | None = None) -> OcrEngine | None:
    """Instantiate an OCR engine; returns ``None`` (with a warning) if it is unavailable.

    ``name`` is a registered engine (``tesseract``, ``azure``, anything added with
    :func:`register_ocr_engine`), ``none``, or ``package.module:Class`` for an engine that
    lives outside this package.
    """
    name = (name or "none").strip()
    if name.lower() == "none":
        return None
    if ":" in name:
        factory = _load_factory(name)
    elif name.lower() in _REGISTRY:
        factory = _REGISTRY[name.lower()]
    else:
        raise ValueError(f"Unknown OCR engine {name!r}; expected one of {available_ocr_engines()} "
                         "or 'package.module:Class'")
    try:
        return factory(params, settings or get_settings())
    except OcrUnavailableError as exc:
        log.warning("OCR engine '%s' unavailable: %s", name, exc)
        return None


# --------------------------------------------------------------------------- #
# Cell-level OCR strategies (paper: single-cell / multi-cell)
# --------------------------------------------------------------------------- #
def crop_cell(image: np.ndarray, bbox: BBox, shrink: int) -> np.ndarray:
    """Crop a cell shrunk by ``shrink`` px per side and binarise it for OCR."""
    h, w = image.shape[:2]
    b = BBox(bbox.x0 + shrink, bbox.y0 + shrink, bbox.x1 - shrink, bbox.y1 - shrink).clip(w, h).to_int()
    if b.width <= 1 or b.height <= 1:
        return np.full((8, 8, 3), 255, np.uint8)
    crop = image[int(b.y0):int(b.y1), int(b.x0):int(b.x1)]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 15)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def _stack_with_separators(crops: Sequence[np.ndarray], separator: np.ndarray) -> np.ndarray:
    parts: list[np.ndarray] = []
    for i, crop in enumerate(crops):
        parts.append(crop)
        if i < len(crops) - 1:
            parts.append(separator)
    width = max(p.shape[1] for p in parts)
    padded = []
    for p in parts:
        canvas = np.full((p.shape[0], width, 3), 255, np.uint8)
        canvas[:, : p.shape[1]] = p
        padded.append(canvas)
    return np.vstack(padded)


def _separator_regex(marker: str) -> re.Pattern[str]:
    # Tolerate OCR inserting spaces/newlines inside the marker or dropping the '@'.
    core = r"\s*".join(re.escape(ch) for ch in marker.rstrip("@"))
    return re.compile(core + r"\s*@?", re.IGNORECASE)


def ocr_cells(
    image: np.ndarray,
    cells: Sequence[GridCell],
    engine: OcrEngine,
    params: OcrParams,
    shrink: int,
) -> dict[int, str]:
    """OCR the cells of one table with the configured cell-level strategy."""
    scope = params.scope
    if scope is OcrScope.AUTO:
        scope = OcrScope.SINGLE if len(cells) < params.auto_threshold else OcrScope.MULTI
    log.debug("Cell OCR: %d cells, strategy=%s, engine=%s", len(cells), scope.value, engine.name)

    def single(cell: GridCell) -> str:
        return engine.image_to_text(crop_cell(image, cell.bbox, shrink)).replace("\n", " ").strip()

    if scope is not OcrScope.MULTI:
        return {c.id: single(c) for c in cells}

    separator = cv2.imread(str(params.row_separator_image))
    if separator is None:
        log.warning("Row-separator image missing (%s); using single-cell OCR", params.row_separator_image)
        return {c.id: single(c) for c in cells}
    pattern = _separator_regex(params.row_separator_text)

    results: dict[int, str] = {}
    columns: dict[int, list[GridCell]] = {}
    for cell in cells:
        if cell.row_span > 1 or cell.col_span > 1:
            results[cell.id] = single(cell)  # merged cells are always read on their own
        else:
            columns.setdefault(cell.col, []).append(cell)

    for col_cells in columns.values():
        col_cells.sort(key=lambda c: c.row)
        for start in range(0, len(col_cells), params.chunk_size):
            chunk = col_cells[start:start + params.chunk_size]
            if len(chunk) == 1:
                results[chunk[0].id] = single(chunk[0])
                continue
            stacked = _stack_with_separators([crop_cell(image, c.bbox, shrink) for c in chunk], separator)
            pieces = [p.replace("\n", " ").strip() for p in pattern.split(engine.image_to_text(stacked))]
            if len(pieces) != len(chunk):
                log.debug("Multi-cell sanity check failed (%d != %d); falling back", len(pieces), len(chunk))
                pieces = [single(c) for c in chunk]
            for cell, text in zip(chunk, pieces):
                results[cell.id] = text
    return results


# --------------------------------------------------------------------------- #
# OCR auto-correction (column-trend based)
# --------------------------------------------------------------------------- #
_TO_DIGIT = {"O": "0", "o": "0", "D": "0", "Q": "0", "I": "1", "l": "1", "|": "1", "!": "1", "S": "5", "B": "8", "Z": "2"}
_TO_LETTER = {"0": "O", "1": "I", "|": "I", "!": "I", "5": "S", "8": "B", "2": "Z", "$": "S"}
_CODE_RE = re.compile(r"^[A-Z0-9|!$\-_/.#+]+$")


def _char_class(ch: str) -> str:
    if ch.isdigit():
        return "d"
    if ch.isalpha():
        return "a"
    return "o"


def correct_ocr_column(values: Sequence[str]) -> list[str]:
    """Fix typical OCR confusions using the trend of a column.

    Only *code-like* columns (part numbers, IDs: no spaces, upper-case letters
    and digits) are touched. For every character position the dominant
    character class (digit vs. letter) across the column is computed, and
    ambiguous glyphs (``O``/``0``, ``I``/``1``/``|``, ``S``/``5``, ...) are
    mapped to the dominant class. Stray border glyphs (``|``, ``!``) at the start
    or end of a cell are removed.
    """
    cleaned = [re.sub(r"^[|!]+\s*|\s*[|!]+$", "", v or "") for v in values]
    tokens = [v for v in cleaned if v]
    if len(tokens) < 3:
        return cleaned
    code_like = sum(1 for v in tokens if _CODE_RE.match(v)) / len(tokens)
    if code_like < 0.8:
        return cleaned

    max_len = max(len(v) for v in tokens)
    dominant: list[str] = []
    for pos in range(max_len):
        counts = Counter(_char_class(v[pos]) for v in tokens if pos < len(v))
        cls, votes = counts.most_common(1)[0] if counts else ("", 0)
        total = sum(counts.values())
        dominant.append(cls if cls in ("a", "d") and votes >= 2 and votes / total >= 0.6 else "")

    corrected = []
    for v in cleaned:
        chars = list(v)
        for pos, ch in enumerate(chars):
            target = dominant[pos] if pos < len(dominant) else ""
            if target == "d" and ch in _TO_DIGIT and not ch.isdigit():
                chars[pos] = _TO_DIGIT[ch]
            elif target == "a" and ch in _TO_LETTER and not ch.isalpha():
                chars[pos] = _TO_LETTER[ch]
        corrected.append("".join(chars))
    return corrected
