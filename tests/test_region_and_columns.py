import numpy as np

from spartan.column_detection import find_vertical_gaps, segment_by_gaps
from spartan.models import BBox, SegmentKind
from spartan.pre_processing import preprocess_image
from spartan.region_segmentation import find_boundary_tables, find_text_tables, segment_region

from .conftest import draw_grid_table, make_word


def _page_with_table():
    page = np.full((2000, 1600, 3), 255, np.uint8)
    table = draw_grid_table(rows=[0, 80, 160, 240, 320], cols=[0, 400, 800, 1200])
    page[800:800 + table.shape[0], 200:200 + table.shape[1]] = table
    return page


def test_boundary_detector_finds_table(config):
    page = _page_with_table()
    _, binary = preprocess_image(page, config)
    tables, cells = find_boundary_tables(binary, config)
    assert len(tables) == 1
    t = tables[0]
    assert abs(t.x0 - 200) < 10 and abs(t.y0 - 800) < 10 and abs(t.x1 - 1400) < 10
    assert len(cells) >= 12


def test_text_detector_finds_borderless_table(config):
    words = []
    for r, (a, b, c) in enumerate([("Part", "Package", "Status"), ("AD1328", "LQFP", "EOL"),
                                    ("AD1401", "QFN", "EOL"), ("SSM2141", "SOIC", "Active")]):
        y = 300 + r * 60
        words += [make_word(a, 200, y), make_word(b, 700, y), make_word(c, 1100, y)]
    cands = find_text_tables(words, 1600, 2000, config)
    assert len(cands) == 1
    assert len(cands[0].pseudo_columns) == 2          # three columns
    assert len(cands[0].pseudo_rows) == 3             # four rows


def test_numbered_list_is_not_a_table(config):
    words = []
    for r in range(4):
        y = 300 + r * 60
        words += [make_word(f"{r + 1}.", 200, y, 30), make_word("Some sentence text here", 400, y)]
    assert find_text_tables(words, 1600, 2000, config) == []


def test_segmentation_order(config):
    page = _page_with_table()
    words = [make_word("Notice", 200, 200), make_word("Footer", 200, 1700)]
    segments, _, _ = segment_region(page, words, config)
    assert [s.kind for s in segments] == [SegmentKind.PRE_TEXT, SegmentKind.TABLE, SegmentKind.POST_TEXT]


def test_column_gap_detection(config):
    ink = np.zeros((3000, 2400), np.uint8)
    for y in range(300, 2700, 60):           # two text columns separated by a 120 px gutter
        ink[y:y + 30, 200:1140] = 255
        ink[y:y + 30, 1260:2200] = 255
    gaps = find_vertical_gaps(ink, config)
    assert len(gaps) == 1 and 1140 <= gaps[0].x <= 1260
    boxes = segment_by_gaps(2400, 3000, gaps)
    assert len(boxes) >= 2
    assert all(isinstance(b, BBox) for b in boxes)
