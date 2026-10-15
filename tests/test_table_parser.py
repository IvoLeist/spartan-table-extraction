from spartan.models import MergeType
from spartan.table_parser import parse_table

from .conftest import draw_grid_table


def test_regular_grid(config):
    img = draw_grid_table(rows=[0, 60, 120, 180], cols=[0, 300, 600, 900])
    table = parse_table(img, [], config)
    assert (table.n_rows, table.n_cols) == (3, 3)
    assert len(table.cells) == 9
    assert all(c.merge_type is MergeType.NONE for c in table.cells)


def test_spanning_cells(config):
    # Row line 1 missing in column 0 -> cell (0,0) spans two rows.
    # Column line 2 missing in row 2 -> cell (2,1) spans two columns.
    img = draw_grid_table(
        rows=[0, 60, 120, 180], cols=[0, 300, 600, 900], skip_h={(1, 0)}, skip_v={(2, 2)},
    )
    table = parse_table(img, [], config)
    spans = {(c.row, c.col): (c.row_span, c.col_span) for c in table.cells}
    assert spans[(0, 0)] == (2, 1)
    assert spans[(2, 1)] == (1, 2)
    assert len(table.cells) == 7


def test_header_candidate(config):
    # Full-width title row: vertical lines missing in row 0.
    img = draw_grid_table(rows=[0, 60, 120, 180], cols=[0, 300, 600, 900], skip_v={(1, 0), (2, 0)})
    table = parse_table(img, [], config)
    assert len(table.header_candidates) == 1
    assert table.header_candidates[0].col_span == 3
