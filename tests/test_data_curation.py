import numpy as np

from spartan.data_curation import build_tables
from spartan.models import BBox, Segment, SegmentKind, TableType
from spartan.table_parser import parse_table

from .conftest import draw_grid_table


def _curate(config, img, texts_by_pos, method="boundary"):
    parsed = parse_table(img, [], config)
    texts = {c.id: texts_by_pos.get((c.row, c.col), "") for c in parsed.cells}
    seg = Segment("table_1", SegmentKind.TABLE, BBox(0, 0, img.shape[1], img.shape[0]), img, [], method)
    return build_tables(parsed, texts, seg, 1, BBox(0, 0, 0, 0), config, "pdf_text_layer", False)


def test_normal_table_with_header(config):
    img = draw_grid_table(rows=[0, 60, 120, 180], cols=[0, 300, 600, 900])
    texts = {(0, 0): "Model", (0, 1): "Replacement", (0, 2): "Pin to Pin",
             (1, 0): "ADAU1328BSTZ", (1, 1): "AD1938WBSTZ", (1, 2): "Yes",
             (2, 0): "ADAU1401YSTZ", (2, 1): "ADAU1401AWBSTZ", (2, 2): "No"}
    [table] = _curate(config, img, texts)
    assert table.type is TableType.NORMAL
    assert table.columns == ["Model", "Replacement", "Pin to Pin"]
    assert table.rows == [["ADAU1328BSTZ", "AD1938WBSTZ", "Yes"], ["ADAU1401YSTZ", "ADAU1401AWBSTZ", "No"]]


def test_nested_header_and_row_span(config):
    # Row 0: "Discontinued" spans cols 0-1, "New" spans col 2-3; row 1: four child headers.
    # Column 3 data rows 2-3 are one merged cell.
    img = draw_grid_table(
        rows=[0, 60, 120, 180, 240], cols=[0, 250, 500, 750, 1000],
        skip_v={(1, 0), (3, 0)}, skip_h={(3, 3)},
    )
    texts = {(0, 0): "Discontinued", (0, 2): "New",
             (1, 0): "Part", (1, 1): "Package", (1, 2): "Part", (1, 3): "Package",
             (2, 0): "BSC016N03LS", (2, 1): "TDSON-8", (2, 2): "ISC016N03", (2, 3): "TDSON-8",
             (3, 0): "BSC025N03LS", (3, 1): "TDSON-8", (3, 2): "ISC024N03"}
    [table] = _curate(config, img, texts)
    assert table.type is TableType.NESTED
    assert table.header_rows == 2
    assert table.columns == ["Discontinued Part", "Discontinued Package", "New Part", "New Package"]
    assert table.header == [["Discontinued", "Discontinued", "New", "New"], ["Part", "Package", "Part", "Package"]]
    assert table.rows[1] == ["BSC025N03LS", "TDSON-8", "ISC024N03", "TDSON-8"]  # merged value repeated
    assert any(m.row_span == 2 and m.col == 3 for m in table.merged_cells)


def test_sliver_columns_folded(config):
    # A shaded header draws lines 25 px inside the frame; the data rows span across them.
    img = draw_grid_table(
        rows=[0, 60, 120, 180], cols=[0, 25, 325, 625, 925, 950],
        skip_v={(1, 1), (1, 2), (4, 1), (4, 2)},
    )
    texts = {(0, 1): "Name", (0, 2): "Qty", (0, 3): "Price",
             (1, 0): "Alpha", (1, 2): "5", (1, 3): "1.00",
             (2, 0): "Beta", (2, 2): "7", (2, 3): "2.50"}
    [table] = _curate(config, img, texts)
    assert table.columns == ["Name", "Qty", "Price"]
    assert table.header == [["Name", "Qty", "Price"]]
    assert table.rows == [["Alpha", "5", "1.00"], ["Beta", "7", "2.50"]]


def test_currency_columns_join_amounts(config):
    img = draw_grid_table(rows=[0, 60, 120, 180], cols=[0, 400, 500, 750, 850, 1100])
    texts = {(0, 0): "Assets", (0, 2): "2011", (0, 4): "2010",
             (1, 0): "Fund Balance", (1, 1): "$", (1, 2): "12,175,088", (1, 3): "$", (1, 4): "12,458,688",
             (2, 0): "Advances", (2, 2): "69,228", (2, 4): "9,782"}
    [table] = _curate(config, img, texts)
    assert table.columns == ["Assets", "2011", "2010"]
    assert table.rows == [["Fund Balance", "$ 12,175,088", "$ 12,458,688"], ["Advances", "69,228", "9,782"]]


def test_one_value_per_row_text_table_rejected(config):
    # Chart axis ticks picked up by the borderless detector.
    img = draw_grid_table(rows=[0, 60, 120, 180, 240], cols=[0, 300, 600])
    texts = {(0, 0): "5", (0, 1): "5", (1, 0): "0", (1, 1): "0", (2, 0): "-5", (2, 1): "-5", (3, 0): "-10", (3, 1): "-10"}
    assert _curate(config, img, texts, method="text") == []
    assert _curate(config, img, texts) != []


def test_title_row_and_key_value(config):
    # Full-width title row, then a 2-column key/value body with colon-terminated keys.
    img = draw_grid_table(rows=[0, 60, 120, 180], cols=[0, 300, 900], skip_v={(1, 0)})
    texts = {(0, 0): "Forecasted Key Milestones", (1, 0): "Last Order Date:", (1, 1): "2024-09-30",
             (2, 0): "Last Delivery Date:", (2, 1): "2025-03-31"}
    [table] = _curate(config, img, texts)
    assert table.title == "Forecasted Key Milestones"
    assert table.type is TableType.KEY_VALUE
    assert table.pairs()[0] == {"key": "Last Order Date:", "value": "2024-09-30"}


def test_headerless_list_table(config):
    img = draw_grid_table(rows=[0, 60, 120, 180], cols=[0, 300, 600])
    texts = {(0, 0): "HVA-451+", (0, 1): "TB-MVA-2000+", (1, 0): "MVA-2000+", (1, 1): "TB-RVA-2000+",
             (2, 0): "RVA-3000+", (2, 1): "ZX60-3018+"}
    [table] = _curate(config, img, texts)
    assert table.header_rows == 0
    assert table.columns == ["column_1", "column_2"]
    assert len(table.rows) == 3


def test_repeated_header_row_dropped(config):
    img = draw_grid_table(rows=[0, 60, 120, 180, 240], cols=[0, 300, 600])
    texts = {(0, 0): "Model", (0, 1): "Family", (1, 0): "AD1", (1, 1): "F1",
             (2, 0): "Model", (2, 1): "Family", (3, 0): "AD2", (3, 1): "F2"}
    [table] = _curate(config, img, texts)
    assert table.rows == [["AD1", "F1"], ["AD2", "F2"]]


def test_single_box_is_not_a_table(config):
    img = np.full((200, 800, 3), 255, np.uint8)
    img = draw_grid_table(rows=[0, 100], cols=[0, 800])
    assert _curate(config, img, {(0, 0): "Some boxed paragraph"}) == []


def test_form_grid_becomes_key_value(config):
    img = draw_grid_table(rows=[0, 60, 120], cols=[0, 200, 500, 700, 1000], skip_v={(2, 1), (3, 1)})
    texts = {(0, 0): "PDN Number:", (0, 1): "2023070700", (0, 2): "PDN Date:", (0, 3): "July 07, 2023",
             (1, 0): "Title :", (1, 1): "Discontinuance of Select Devices"}
    [table] = _curate(config, img, texts)
    assert table.type is TableType.KEY_VALUE
    assert ["PDN Date:", "July 07, 2023"] in table.rows
