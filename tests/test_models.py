from spartan.models import BBox, group_words_into_lines, words_in_box, words_to_text

from .conftest import make_word


def test_bbox_geometry():
    a, b = BBox(0, 0, 10, 10), BBox(5, 5, 20, 20)
    assert a.intersects(b)
    assert a.intersection(b) == BBox(5, 5, 10, 10)
    assert a.union(b) == BBox(0, 0, 20, 20)
    assert a.overlap_ratio(b) == 0.25
    assert not a.intersects(BBox(10, 0, 20, 10))  # touching edges do not overlap
    assert BBox(0, 0, 10, 10).expand(2) == BBox(-2, -2, 12, 12)


def test_line_grouping_and_text():
    words = [make_word("world", 120, 2), make_word("Hello", 0, 0), make_word("Next", 0, 60)]
    lines = group_words_into_lines(words)
    assert [[w.text for w in ln] for ln in lines] == [["Hello", "world"], ["Next"]]
    assert words_to_text(words) == "Hello world\nNext"


def test_words_in_box_relative():
    words = [make_word("in", 50, 50, 20, 10), make_word("out", 500, 500, 20, 10)]
    inside = words_in_box(words, BBox(40, 40, 100, 100))
    assert [w.text for w in inside] == ["in"]
    assert inside[0].bbox.x0 == 10
