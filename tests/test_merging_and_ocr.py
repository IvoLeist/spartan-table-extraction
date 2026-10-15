from spartan.config import CurationParams
from spartan.models import Table, TableType
from spartan.ocr import correct_ocr_column
from spartan.table_merging import can_merge, merge_tables


def _table(header, rows, widths=(300.0, 300.0), title="", header_rows=1):
    return Table(type=TableType.NORMAL, columns=list(header) or ["column_1", "column_2"], rows=[list(r) for r in rows],
                 column_widths=list(widths), title=title, title_source="table" if title else "none",
                 header_rows=header_rows, raw_header=list(header))


def test_cross_page_continuation_with_repeated_header():
    p = CurationParams()
    a = _table(["Model", "Family"], [["AD1", "F1"]])
    b = _table(["Model", "Family"], [["AD2", "F2"]])
    assert can_merge(a, b, p, 1.0, same_page=False)
    merge_tables(a, b, p)
    assert a.rows == [["AD1", "F1"], ["AD2", "F2"]]


def test_continuation_without_header_keeps_first_row():
    p = CurationParams()
    a = _table(["Model", "Family"], [["AD1", "F1"]])
    b = _table(["AD2", "F2"], [["AD3", "F3"], ["Model", "Family"], ["AD4", "F4"]])
    assert can_merge(a, b, p, 1.0, same_page=False)
    merge_tables(a, b, p)
    assert a.rows == [["AD1", "F1"], ["AD2", "F2"], ["AD3", "F3"], ["AD4", "F4"]]


def test_conflicting_titles_do_not_merge():
    p = CurationParams()
    a = _table(["Part", "Qty"], [["A", "1"]], title="EOL devices with replacement")
    b = _table(["Part", "Qty"], [["B", "2"]], title="EOL devices without replacement")
    assert not can_merge(a, b, p, 1.0, same_page=True)
    assert not can_merge(a, b, p, 1.0, same_page=False)


def test_ocr_column_correction():
    column = ["BSC016N03LS", "BSCO25N03LS", "BSC057N03LS", "|BSC100N03LS"]
    fixed = correct_ocr_column(column)
    assert fixed[1] == "BSC025N03LS"
    assert fixed[3] == "BSC100N03LS"
    # Prose is left alone.
    assert correct_ocr_column(["Some text", "More text", "Other"]) == ["Some text", "More text", "Other"]
