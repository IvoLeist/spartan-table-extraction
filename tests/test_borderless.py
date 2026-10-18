"""Text-based (borderless) detector: false positives, fragmentation and grid."""

from spartan.models import group_words_into_lines
from spartan.region_segmentation import _is_prose, _line_phrases, find_text_tables
from spartan.table_parser import grid_from_separators

from .conftest import make_word


def _phrase(text: str, x: float, y: float) -> list:
    """One word per token, separated by a normal space (they form a single phrase)."""
    words = []
    for token in text.split():
        words.append(make_word(token, x, y))
        x += 18 * len(token) + 10
    return words


def test_two_column_prose_is_not_a_table(config):
    words = []
    for r in range(6):
        y = 300 + r * 45
        words += _phrase("the quick brown fox jumps over", 200, y)
        words += _phrase("a lazy dog sleeps in the sun", 1000, y)
    assert find_text_tables(words, 1600, 2000, config) == []


def test_prose_check_ignores_empty_columns(config):
    words = []
    for r in range(6):
        y = 300 + r * 45
        words += _phrase("the quick brown fox jumps over", 200, y)
        words += _phrase("a lazy dog sleeps in the sun", 1000, y)
    phrases = _line_phrases(group_words_into_lines(words), config)
    assert _is_prose(phrases, [100, 180, 900, 1800])      # first column holds no text


def test_symbol_bullets_are_a_list(config):
    words = []
    for r in range(4):
        y = 300 + r * 60
        words += [make_word("\u0083", 200, y, 20), make_word("holiday pay", 400, y)]
    assert find_text_tables(words, 1600, 2000, config) == []


def test_rotated_chart_labels_are_not_a_table(config):
    words = []
    for r in range(4):
        y = 300 + r * 60
        words += [make_word(ch, 200 + i * 300, y) for i, ch in enumerate("abcd")]
    assert find_text_tables(words, 1600, 2000, config) == []


def test_close_numeric_columns_are_split(config):
    # 30 px between the numbers: below the phrase gap, but clearly more than a space.
    words = []
    for r, label in enumerate(["Receipts", "Taxes", "Transfers", "Total"]):
        y = 300 + r * 60
        words += [make_word(label, 200, y)]
        words += [make_word(f"{r}1.0", 1000 + c * 120, y, 90) for c in range(3)]
    cands = find_text_tables(words, 1600, 2000, config)
    assert len(cands) == 1
    assert len(cands[0].pseudo_columns) == 3          # label + three number columns


def test_section_heading_does_not_split_table(config):
    words, y = [], 300
    for label in ["Receipts", "Taxes", "Transfers", "HEADING", "Defense", "Interest", "Subsidies"]:
        if label == "HEADING":
            words += [make_word("Expenditures", 600, y)]
        else:
            words += [make_word(label, 200, y), make_word("12.5", 1000, y, 72), make_word("13.5", 1200, y, 72)]
        y += 60 if label != "HEADING" else 75
    cands = find_text_tables(words, 1600, 2000, config)
    assert len(cands) == 1
    assert cands[0].bbox.y0 < 300 and cands[0].bbox.y1 > y - 75


def test_tables_with_same_layout_stay_separate(config):
    words, y = [], 300
    for _table in range(2):
        for label in ["Obligated", "Available", "Unavailable"]:
            words += [make_word(label, 200, y), make_word("1,000", 1000, y, 90), make_word("2,000", 1200, y, 90)]
            y += 60
        y += 75
    assert len(find_text_tables(words, 1600, 2000, config)) == 2


def test_statement_sections_join_across_headings(config):
    # Balance-sheet sections: same right-aligned amount columns, two headings and a wide gap between.
    words, y = [], 300
    sections = [["Treasury", "Receivable", "Advances"], ["Payable", "Benefits", "Leave"], ["Unexpended", "Cumulative", "Total"]]
    for i, labels in enumerate(sections):
        if i:
            y += 30
            words += [make_word("Liabilities", 200, y)]
            y += 60
            words += [make_word("Intragovernmental", 200, y), make_word("Liabilities", 530, y)]
            y += 75
        for label in labels:
            words += [make_word(label, 230, y), make_word("12,175", 1000, y, 108), make_word("9,782", 1210, y, 90)]
            y += 45
    cands = find_text_tables(words, 1600, 2400, config)
    assert len(cands) == 1
    assert cands[0].bbox.y0 < 300 and cands[0].bbox.y1 > y - 45


def test_indented_row_labels_stay_in_the_table(config):
    words, y = [], 300
    for label, x in [("Research", 160), ("Gross", 220), ("Net", 160), ("Education", 220), ("Less", 160), ("Total", 220)]:
        words += [make_word(label, x, y), make_word("836,755", 1000, y, 126), make_word("775,422", 1200, y, 126)]
        y += 45
    [cand] = find_text_tables(words, 1600, 2000, config)
    assert cand.bbox.x0 < 160


def test_grid_from_separators(config):
    table = grid_from_separators(900, 300, [300, 600], [100, 200], config)
    assert (table.n_rows, table.n_cols) == (3, 3)
    assert len(table.cells) == 9


# --------------------------------------------------------------------------- #
# Tables that rule only the header
# --------------------------------------------------------------------------- #
def _body_words(n_rows: int, numeric: bool = True) -> list:
    words = [make_word("Country", 20, 40), make_word("Salary", 320, 40), make_word("Tax", 620, 40)]
    for i in range(n_rows):
        y = 120 + i * 60
        values = (f"{i}2.5", f"{i}7.1") if numeric else ("review", "annual")
        words += [make_word(f"Name{chr(65 + i)}", 20, y), make_word(values[0], 320, y), make_word(values[1], 620, y)]
    return words


def test_unruled_body_rows_are_split(config):
    from spartan.table_parser import split_unruled_rows

    table = grid_from_separators(900, 120 + 8 * 60, [300, 600], [100], config)   # header line only
    split = split_unruled_rows(table, _body_words(8), config)
    assert (split.n_rows, split.n_cols) == (9, 3)


def test_wrapped_text_rows_are_not_split(config):
    from spartan.table_parser import split_unruled_rows

    table = grid_from_separators(900, 120 + 8 * 60, [300, 600], [100], config)
    assert split_unruled_rows(table, _body_words(8, numeric=False), config).n_rows == 2


def test_side_by_side_tables_are_separated(config):
    import numpy as np

    from spartan.pre_processing import preprocess_image
    from spartan.region_segmentation import find_boundary_tables

    from .conftest import draw_grid_table

    page = np.full((1500, 2000, 3), 255, np.uint8)
    grid = draw_grid_table(rows=[0, 80, 160, 240, 320], cols=[0, 400, 800])
    page[300:300 + grid.shape[0], 100:100 + grid.shape[1]] = grid
    page[300:300 + grid.shape[0], 1100:1100 + grid.shape[1]] = grid
    _, binary = preprocess_image(page, config)
    tables, _ = find_boundary_tables(binary, config)
    assert len(tables) == 2


def test_ruled_header_joins_text_body(config):
    from spartan.models import BBox
    from spartan.region_segmentation import unify_tables

    words = []
    for r in range(8):
        y = 400 + r * 60
        words += [make_word(f"Name{r}", 200, y), make_word(f"{r}1.5", 800, y, 72), make_word(f"{r}2.5", 1200, y, 72)]
    words += [make_word("Country", 200, 320), make_word("Salary", 800, 320), make_word("Tax", 1200, 320)]
    body = find_text_tables([w for w in words if w.bbox.y0 >= 400], 1600, 2000, config)
    header = BBox(180, 300, 1300, 395)
    tables = unify_tables([header], body, words, 1600, 2000, config)
    assert [t.method for t in tables] == ["text"]
    assert tables[0].bbox.y0 <= 300


def test_unruled_body_columns_are_split(config):
    from spartan.table_parser import build_grid, split_unruled_columns

    # Rows ruled everywhere, column lines only in the header row.
    ys = [0, 100, 160, 220, 280, 340]
    h = [(0.0, y, 899.0, y) for y in ys[1:-1]]
    v = [(300, 0.0, 300, 100.0), (600, 0.0, 600, 100.0)]
    table = build_grid(h, v, 900, 341, config)
    assert table.n_cols == 3 and len(table.cells) == 3 + 4      # body rows are single cells
    words = []
    for r in range(4):
        y = 115 + r * 60
        words += [make_word(f"Name{chr(65 + r)}", 20, y), make_word(f"{r}1.5", 320, y), make_word(f"{r}2.5", 620, y)]
    split = split_unruled_columns(table, words, config)
    assert len(split.cells) == 15


def test_section_title_rows_keep_spanning(config):
    from spartan.table_parser import build_grid, split_unruled_columns

    ys = [0, 100, 160, 220, 280, 340]
    h = [(0.0, y, 899.0, y) for y in ys[1:-1]]
    v = [(300, 0.0, 300, 100.0), (600, 0.0, 600, 100.0)]
    table = build_grid(h, v, 900, 341, config)
    words = [make_word("Affected products and packages", 20, 115 + r * 60) for r in range(4)]
    assert split_unruled_columns(table, words, config) is table
