"""Writers for the final outputs (JSON document, CSV per table)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .models import Table


def public_document(document: dict[str, Any]) -> dict[str, Any]:
    """Strip internal keys (prefixed with ``_``) before serialisation."""
    return {k: v for k, v in document.items() if not k.startswith("_")}


def write_json(data: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(public_document(data), fh, indent=2, ensure_ascii=False)
    return path


def write_table_csv(table: Table, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig so Excel opens the file with the right encoding.
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(table.columns)
        writer.writerows(table.rows)
    return path


def write_csvs(tables: list[Table], folder: Path) -> list[Path]:
    """One CSV per table, named ``table_<n>.csv`` (titles are in the JSON).

    File names are kept short on purpose: long titles easily exceed the
    260-character path limit on Windows.
    """
    return [write_table_csv(table, folder / f"table_{i}.csv") for i, table in enumerate(tables, start=1)]
