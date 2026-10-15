"""Shared fixtures: synthetic table images and the optional real-PDF dataset."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from spartan.config import PipelineConfig
from spartan.models import BBox, Word

DATASET_DIR = Path(os.environ.get("SPARTAN_DATASET_DIR") or Path(__file__).resolve().parents[2] / "Dataset")


@pytest.fixture
def config() -> PipelineConfig:
    return PipelineConfig(ocr_engine="none")


@pytest.fixture
def dataset_dir() -> Path:
    if not DATASET_DIR.is_dir():
        pytest.skip(f"dataset not found at {DATASET_DIR}")
    return DATASET_DIR


def make_word(text: str, x: float, y: float, w: float | None = None, h: float = 30) -> Word:
    w = w if w is not None else 18 * len(text)
    return Word(text, BBox(x, y, x + w, y + h))


def draw_grid_table(
    rows: list[float], cols: list[float], skip_h: set[tuple[int, int]] = frozenset(),
    skip_v: set[tuple[int, int]] = frozenset(), thickness: int = 3, margin: int = 0,
) -> np.ndarray:
    """Draw a bordered grid. ``skip_h`` holds (row_line_index, col) segments to leave out
    (creates row-spanning cells); ``skip_v`` holds (col_line_index, row) segments
    (creates column-spanning cells)."""
    width, height = int(cols[-1]) + margin + 1, int(rows[-1]) + margin + 1
    img = np.full((height, width, 3), 255, np.uint8)
    for ri, y in enumerate(rows):
        for ci in range(len(cols) - 1):
            if (ri, ci) not in skip_h:
                cv2.line(img, (int(cols[ci]), int(y)), (int(cols[ci + 1]), int(y)), (0, 0, 0), thickness)
    for ci, x in enumerate(cols):
        for ri in range(len(rows) - 1):
            if (ci, ri) not in skip_v:
                cv2.line(img, (int(x), int(rows[ri])), (int(x), int(rows[ri + 1])), (0, 0, 0), thickness)
    return img
