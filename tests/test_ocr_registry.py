"""OCR engine selection: built-ins, registered engines and module:Class specs."""

import pytest

from spartan.config import OcrParams, Settings
from spartan.ocr import OcrUnavailableError, available_ocr_engines, create_ocr_engine, register_ocr_engine


class DummyEngine:
    name = "dummy"

    def __init__(self, params, settings):
        self.params = params

    def image_to_words(self, image):
        return []

    def image_to_text(self, image):
        return ""


class BrokenEngine(DummyEngine):
    def __init__(self, params, settings):
        raise OcrUnavailableError("not installed")


def test_registered_engine_is_selectable():
    register_ocr_engine("dummy", DummyEngine)
    assert "dummy" in available_ocr_engines()
    assert isinstance(create_ocr_engine("Dummy", OcrParams(), Settings()), DummyEngine)


def test_module_class_spec():
    engine = create_ocr_engine(f"{__name__}:DummyEngine", OcrParams(), Settings())
    assert engine.name == "dummy"


def test_unavailable_engine_degrades_to_none():
    assert create_ocr_engine(f"{__name__}:BrokenEngine", OcrParams(), Settings()) is None
    assert create_ocr_engine("none", OcrParams(), Settings()) is None


def test_missing_pytesseract_degrades_to_none(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "pytesseract", None)    # makes `import pytesseract` fail
    assert create_ocr_engine("tesseract", OcrParams(), Settings()) is None


def test_unknown_engine_is_an_error():
    with pytest.raises(ValueError):
        create_ocr_engine("no-such-engine", OcrParams(), Settings())
    with pytest.raises(ValueError):
        create_ocr_engine("no_such_module:Engine", OcrParams(), Settings())
