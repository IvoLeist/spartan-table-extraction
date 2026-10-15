from spartan.config import Settings
from spartan.plugins.pcn import map_to_pcn_schema, match_score, normalize_date


def test_normalize_date():
    assert normalize_date("20-Sep-2023") == "2023-09-20"
    assert normalize_date("September 01, 2023") == "2023-09-01"
    assert normalize_date("Sept 15, 2024") == "2024-09-15"
    assert normalize_date("4/25/2024") == "2024-04-25"
    assert normalize_date("Nov 19th, 2024") == "2024-11-19"
    assert normalize_date("Available Now") == "Available Now"


def test_match_score():
    assert match_score("Discontinued OPN", "discontinued opn") == 1.0
    assert match_score("NEW (REPLACEMENT) OPN", "Replacement OPN") > 0.7
    assert match_score("Sample Date", "Date") == 0.0          # generic single word
    assert match_score("Product Family", "Product") == 0.0


def test_deterministic_mapping_without_llm():
    document = {
        "full_text": "Publication Date: 20-Sep-2023\nAnalog Devices, Inc. One Analog Way",
        "tables": [
            {"id": "table_1", "type": "key_value", "columns": ["key", "value"], "header_rows": 0,
             "rows": [["Last Time Buy Date:", "20-Sep-2024"], ["Last Time Ship Date:", "20-Mar-2025"]], "title": ""},
            {"id": "table_2", "type": "normal", "columns": ["Model", "Replacement Part", "Pin To Pin Compatible"],
             "header_rows": 1, "title": "Discontinued Parts",
             "rows": [["ADAU1328BSTZ", "AD1938WBSTZ", "Yes"], ["SSM2141PZ", "None", "N/A"]]},
            {"id": "table_3", "type": "normal", "columns": ["column_1", "column_2"], "header_rows": 0,
             "title": "Affected Part Numbers", "rows": [["AQ7520-08UTG", "SP3420-04UTG"]]},
        ],
    }
    out = map_to_pcn_schema(document, settings=Settings(llm_provider="none"), use_llm=False)
    assert out["PCN_DATE"] == "2023-09-20"
    assert out["Last_Time_Buy_Date"] == "2024-09-20"
    assert out["Last_Shipment_Date"] == "2025-03-20"
    assert out["Manufacturer_Name"] == "Analog Devices, Inc."
    cols = out["tables"][0]["columns"]
    assert cols["Impacted_Part"] == ["ADAU1328BSTZ", "SSM2141PZ", "AQ7520-08UTG", "SP3420-04UTG"]
    assert cols["Replacement_Part"][:2] == ["AD1938WBSTZ", "None"]
    assert len({len(v) for v in cols.values()}) == 1          # columns stay aligned
