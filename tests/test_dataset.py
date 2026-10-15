"""End-to-end regression tests on real PCN documents (skipped when ../Dataset is absent)."""

import json

import pytest
from fastapi.testclient import TestClient

from spartan.cli import main as cli_main
from spartan.config import PipelineConfig
from spartan.pipeline import process_pdf

pytestmark = pytest.mark.dataset


def _tables(dataset_dir, name):
    return process_pdf(dataset_dir / name, PipelineConfig(ocr_engine="none"))["tables"]


def test_analog_devices(dataset_dir):
    tables = _tables(dataset_dir, "Analog Devices.pdf")
    kv = next(t for t in tables if t["type"] == "key_value")
    assert {"key": "Last Time Buy Date:", "value": "20-Sep-2024"} in kv["pairs"]
    parts = next(t for t in tables if t["columns"][:1] == ["Model"])
    assert parts["pages"] == [1, 2]                  # merged across the page break
    assert len(parts["rows"]) == 22                  # repeated header on page 2 removed
    assert parts["rows"][0] == ["ADAU1328BSTZ", "ADAU1328", "AD1938WBSTZ", "Yes", "None"]


def test_infineon_nested_table_across_ten_pages(dataset_dir):
    [table] = _tables(dataset_dir, "Infineon - 1.pdf")
    assert table["type"] == "nested"
    assert table["columns"][2] == "DISCONTINUED OPN"
    assert table["columns"][4] == "NEW (REPLACEMENT) Sales name"
    assert len(table["rows"]) == 118
    assert table["rows"][0][:3] == ["BSC016N03LS G", "SP000237663", "BSC016N03LSGATMA1"]   # exact text, no OCR noise


def test_renesas_dense_table(dataset_dir):
    [table] = _tables(dataset_dir, "Renesas - 2.pdf")
    assert len(table["rows"]) == 147
    assert table["rows"][0][0] == "180M-01LF"


def test_cli_and_api(dataset_dir, tmp_path):
    assert cli_main(["--file", str(dataset_dir / "Analog Devices.pdf"), "--output", str(tmp_path),
                     "--csv", "--ocr-engine", "none", "--log-level", "WARNING"]) == 0
    doc = json.loads((tmp_path / "Analog Devices" / "Analog Devices.json").read_text(encoding="utf-8"))
    assert len(doc["tables"]) == 3
    assert list((tmp_path / "Analog Devices" / "csv").glob("*.csv"))

    from spartan.api import app

    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    with open(dataset_dir / "Analog Devices.pdf", "rb") as fh:
        response = client.post("/extract", files={"file": ("Analog Devices.pdf", fh, "application/pdf")})
    assert response.status_code == 200
    assert len(response.json()["tables"]) == 3
    bad = client.post("/extract", files={"file": ("x.pdf", b"not a pdf", "application/pdf")})
    assert bad.status_code == 400 and "error" in bad.json()
