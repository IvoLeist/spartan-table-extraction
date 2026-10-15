"""PCN / PDN schema mapping (optional plugin).

Maps a SPARTAN document onto the fixed Product-Change-Notification schema in
``assets/fixed_schema.json`` using the keyword examples in
``assets/configuration.json``. This is the paper's optional *LLM hook*, made
robust and usable offline:

1. **Deterministic first.** Table columns, key-value pairs and ``Label: value``
   lines are matched against the configured keywords (exact, containment and
   fuzzy matching). Part-number fields additionally require the column content
   to look like part numbers, and date fields require a parseable date.
2. **LLM only for the gaps.** Unmatched column headers, list-table titles and
   still-empty metadata fields are sent to the configured LLM (local Ollama by
   default) with JSON-schema constrained output and temperature 0.
3. **Normalisation.** Dates are converted to ``YYYY-MM-DD``; table rows stay
   aligned across columns.

The output keeps the format the SPARTAN front-end expects: scalar fields at the
top level and ``tables[0].columns`` holding equally long lists. How every value
was obtained is recorded in ``document["pcn_provenance"]``.
"""

from __future__ import annotations

import copy
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from ..config import ASSETS_DIR, Settings, get_settings
from .llm import LLMClient, create_llm_client

log = logging.getLogger(__name__)

TABLE_KEY = "tables"
IMPACTED_TABLE_HEADERS = "Impacted_Table_Headers"
LIST_FIELD = "Impacted_Part"         # where cells of "affected parts" list tables go
MATCH_THRESHOLD = 0.6


# --------------------------------------------------------------------------- #
# Matching helpers
# --------------------------------------------------------------------------- #
def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^0-9a-z#]+", " ", (text or "").lower())).strip()


def match_score(header: str, synonym: str) -> float:
    """Similarity between a header/label and a configured keyword (0..1)."""
    h, s = _norm(header), _norm(synonym)
    if not h or not s:
        return 0.0
    if h == s:
        return 1.0
    h_tokens, s_tokens = h.split(), s.split()
    # Containment counts only for multi-word keywords or near-complete coverage, so
    # generic single words ("Date", "Product") do not match "Sample Date" / "Product Family".
    if len(s_tokens) <= len(h_tokens) and f" {s} " in f" {h} ":
        coverage = len(s_tokens) / len(h_tokens)
        if len(s_tokens) >= 2 or coverage >= 0.75:
            return 0.5 + 0.4 * coverage
    if 2 <= len(h_tokens) < len(s_tokens) and f" {h} " in f" {s} ":
        return 0.5 + 0.3 * len(h_tokens) / len(s_tokens)
    ratio = SequenceMatcher(None, h, s).ratio()
    return 0.9 * ratio if ratio >= 0.85 else 0.0


def field_scores(header: str, synonyms: dict[str, list[str]]) -> dict[str, float]:
    """Best keyword score per field (only scores above the threshold)."""
    scores = {}
    for field_name, words in synonyms.items():
        score = max((match_score(header, w) for w in words), default=0.0)
        if score >= MATCH_THRESHOLD:
            scores[field_name] = score
    return scores


def best_field(header: str, synonyms: dict[str, list[str]]) -> tuple[str | None, float]:
    scores = field_scores(header, synonyms)
    if not scores:
        return None, 0.0
    name = max(scores, key=scores.get)
    return name, scores[name]


_NULL_VALUES = {"", "none", "n/a", "na", "null", "-", "--", "tbd", "not applicable", "no replacement"}


def looks_like_part_column(values: list[str]) -> bool:
    """Part-number columns mostly hold short codes containing digits."""
    cells = [v.strip() for v in values if v.strip().lower() not in _NULL_VALUES]
    if not cells:
        return True
    codes = sum(1 for v in cells if any(ch.isdigit() for ch in v) and 3 <= len(v) <= 40 and v.count(" ") <= 2)
    return codes / len(cells) >= 0.5


# --------------------------------------------------------------------------- #
# Dates
# --------------------------------------------------------------------------- #
_DATE_FORMATS = (
    "%Y-%m-%d", "%Y/%m/%d", "%d-%b-%Y", "%d-%B-%Y", "%d-%b-%y", "%d %b %Y", "%d %B %Y",
    "%b %d %Y", "%B %d %Y", "%b %d, %Y", "%B %d, %Y", "%m/%d/%Y", "%m/%d/%y", "%d%m%Y", "%Y%m%d",
)
_DATE_PATTERNS = (
    r"\d{4}[-/]\d{1,2}[-/]\d{1,2}",
    r"\d{1,2}[-\s][A-Za-z]{3,9}[-\s]\d{2,4}",
    r"[A-Za-z]{3,9}\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}",
    r"\d{1,2}/\d{1,2}/\d{2,4}",
    r"\d{1,2}\.\d{1,2}\.\d{4}",
)


def normalize_date(value: str) -> str:
    """Return ``YYYY-MM-DD`` for the first date found in ``value``.

    Slash dates are read in US order (month/day/year), as used by most
    component manufacturers. Unparseable values are returned unchanged.
    """
    if not value:
        return value
    for pattern in _DATE_PATTERNS:
        match = re.search(pattern, value)
        if not match:
            continue
        candidate = re.sub(r"(\d)(st|nd|rd|th)", r"\1", match.group()).replace(".", "").strip()
        candidate = re.sub(r"\bSept\b", "Sep", candidate, flags=re.I)
        for fmt in _DATE_FORMATS:
            try:
                return datetime.strptime(candidate, fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
    return value.strip()


def _is_date_field(name: str) -> bool:
    return "date" in name.lower()


def _valid_value(field_name: str, value: str) -> bool:
    """Date fields must contain a parseable date; other fields just need text."""
    if not value or not value.strip():
        return False
    if _is_date_field(field_name):
        return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", normalize_date(value)))
    return True


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class PcnSchema:
    template: dict[str, Any]
    scalar_fields: list[str]
    table_fields: list[str]
    synonyms: dict[str, list[str]]
    list_table_titles: list[str]

    @classmethod
    def load(cls, schema_path: Path | None = None, config_path: Path | None = None) -> "PcnSchema":
        schema_path = schema_path or ASSETS_DIR / "fixed_schema.json"
        config_path = config_path or ASSETS_DIR / "configuration.json"
        template = json.loads(Path(schema_path).read_text(encoding="utf-8"))
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        table_fields = list(template[TABLE_KEY][0]["columns"].keys())
        scalar_fields = [k for k in template if k != TABLE_KEY]
        synonyms = {k: list(v) for k, v in config.items() if k != IMPACTED_TABLE_HEADERS}
        return cls(template, scalar_fields, table_fields, synonyms, list(config.get(IMPACTED_TABLE_HEADERS, [])))

    def empty_output(self) -> dict[str, Any]:
        out = copy.deepcopy(self.template)
        for key in self.scalar_fields:
            out[key] = ""
        for col in self.table_fields:
            out[TABLE_KEY][0]["columns"][col] = []
        return out


# --------------------------------------------------------------------------- #
# Mapper
# --------------------------------------------------------------------------- #
class PcnMapper:
    def __init__(self, schema: PcnSchema, llm: LLMClient | None, settings: Settings):
        self.schema = schema
        self.llm = llm
        self.settings = settings
        self.provenance: dict[str, Any] = {"columns": [], "list_tables": [], "fields": {}, "llm": llm.name if llm else None}

    # ---------------- tables ---------------- #
    @staticmethod
    def _is_part_field(field_name: str) -> bool:
        return field_name.lower().endswith("_part")

    def _map_columns(self, tables: list[dict]) -> dict[int, dict[int, str]]:
        """Return ``{table_index: {column_index: field}}`` for table columns."""
        table_syn = {f: self.schema.synonyms.get(f, []) for f in self.schema.table_fields}
        mapping: dict[int, dict[int, str]] = {}
        unmatched: list[str] = []
        for ti, table in enumerate(tables):
            if table["type"] == "key_value" or table.get("header_rows", 1) == 0:
                continue
            scored = []
            for ci, header in enumerate(table["columns"]):
                scores = field_scores(header, table_syn)
                column = [row[ci] for row in table["rows"] if ci < len(row)]
                for field_name, score in scores.items():
                    if self._is_part_field(field_name) and not looks_like_part_column(column):
                        continue
                    scored.append((score, ci, field_name))
                if not scores:
                    unmatched.append(header)
            mapping[ti] = self._assign(scored, table["columns"])

        if self.llm and unmatched:
            llm_map = self._llm_column_mapping(sorted(set(unmatched)))
            for ti, table in enumerate(tables):
                if ti not in mapping:
                    continue
                taken = set(mapping[ti].values())
                for ci, header in enumerate(table["columns"]):
                    field_name = llm_map.get(header)
                    if not field_name or ci in mapping[ti] or field_name in taken:
                        continue
                    column = [row[ci] for row in table["rows"] if ci < len(row)]
                    if self._is_part_field(field_name) and not looks_like_part_column(column):
                        continue
                    mapping[ti][ci] = field_name
                    taken.add(field_name)
        return mapping

    @staticmethod
    def _assign(scored: list[tuple[float, int, str]], headers: list[str]) -> dict[int, str]:
        """Greedy one-to-one assignment, except for repeated column groups.

        Wrapped list layouts ("EOL Devices | Replacements | EOL Devices | ...")
        repeat the same header; every repetition of the chosen header is kept.
        """
        result: dict[int, str] = {}
        best_header: dict[str, str] = {}
        for score, ci, field_name in sorted(scored, key=lambda t: -t[0]):
            if ci in result:
                continue
            if field_name not in best_header:
                best_header[field_name] = _norm(headers[ci])
                result[ci] = field_name
            elif _norm(headers[ci]) == best_header[field_name]:
                result[ci] = field_name
        return result

    def _llm_column_mapping(self, headers: list[str]) -> dict[str, str]:
        fields = self.schema.table_fields
        examples = {f: self.schema.synonyms.get(f, []) for f in fields}
        prompt = (
            "Map extracted table column headers to the target fields of a product change notification.\n"
            f"Target fields with example headers:\n{json.dumps(examples, indent=2)}\n\n"
            f"Extracted headers:\n{json.dumps(headers, indent=2)}\n\n"
            "For every target field, list the extracted headers that mean the same thing. "
            "A header may be used at most once. Leave a list empty if nothing matches."
        )
        schema = {
            "type": "object",
            "properties": {f: {"type": "array", "items": {"type": "string"}} for f in fields},
            "required": fields,
        }
        result = self.llm.complete_json(prompt, schema) or {}
        mapping = {}
        for field_name, values in result.items():
            if field_name in fields and isinstance(values, list):
                for header in values:
                    if header in headers:
                        mapping[header] = field_name
        return mapping

    def _list_tables(self, tables: list[dict], mapped: dict[int, dict[int, str]]) -> list[int]:
        """Tables without mapped columns whose title says they list affected parts."""
        candidates = [ti for ti, t in enumerate(tables) if t["type"] != "key_value" and not mapped.get(ti)]
        chosen, undecided = [], []
        for ti in candidates:
            title = tables[ti].get("title", "")
            if title and max((match_score(title, h) for h in self.schema.list_table_titles), default=0) >= 0.75:
                chosen.append(ti)
            elif title:
                undecided.append(ti)
        if self.llm and undecided:
            titles = [tables[ti]["title"] for ti in undecided]
            prompt = (
                "Which of these table titles introduce a list of affected / discontinued part numbers?\n"
                f"Examples of such titles:\n{json.dumps(self.schema.list_table_titles, indent=2)}\n\n"
                f"Titles:\n{json.dumps(titles, indent=2)}\n\nReturn the matching titles exactly as given."
            )
            schema = {"type": "object", "properties": {"titles": {"type": "array", "items": {"type": "string"}}},
                      "required": ["titles"]}
            answer = self.llm.complete_json(prompt, schema) or {}
            picked = set(answer.get("titles", []))
            chosen += [ti for ti in undecided if tables[ti]["title"] in picked]
        return sorted(chosen)

    def _fill_table_rows(self, out: dict, tables: list[dict], mapped: dict[int, dict[int, str]], lists: list[int]) -> None:
        columns = out[TABLE_KEY][0]["columns"]
        fields = self.schema.table_fields

        def append(row: dict[str, str]) -> None:
            if any(v.strip() for v in row.values()):
                for f in fields:
                    columns[f].append(row.get(f, ""))

        for ti, table in enumerate(tables):
            col_map = mapped.get(ti) or {}
            if col_map:
                self.provenance["columns"].append({
                    "table": table.get("id"), "mapping": {table["columns"][ci]: f for ci, f in col_map.items()},
                })
                by_field: dict[str, list[int]] = {}
                for ci, f in sorted(col_map.items()):
                    by_field.setdefault(f, []).append(ci)
                blocks = max(len(v) for v in by_field.values())
                for row in table["rows"]:
                    for b in range(blocks):
                        append({
                            f: row[cis[b]] if b < len(cis) else (row[cis[0]] if len(cis) == 1 else "")
                            for f, cis in by_field.items()
                        })
            elif ti in lists and LIST_FIELD in fields:
                self.provenance["list_tables"].append({"table": table.get("id"), "title": table.get("title")})
                for row in table["rows"]:
                    for cell in row:
                        if cell.strip():
                            append({LIST_FIELD: cell.strip()})

    # ---------------- scalar fields ---------------- #
    def _scalar_from_tables(self, tables: list[dict]) -> dict[str, str]:
        scalar_syn = {f: self.schema.synonyms.get(f, []) for f in self.schema.scalar_fields}
        found: dict[str, str] = {}
        for table in tables:
            if table["type"] == "key_value":
                for key, value in table["rows"]:
                    field_name, _ = best_field(key, scalar_syn)
                    if field_name and field_name not in found and _valid_value(field_name, value):
                        found[field_name] = value.strip()
                        self.provenance["fields"][field_name] = f"key-value table {table.get('id')}"
            elif table.get("header_rows", 1):
                for ci, header in enumerate(table["columns"]):
                    field_name, _ = best_field(header, scalar_syn)
                    if field_name and field_name not in found:
                        value = next((r[ci] for r in table["rows"] if ci < len(r) and r[ci].strip()), "")
                        if _valid_value(field_name, value):
                            found[field_name] = value.strip()
                            self.provenance["fields"][field_name] = f"column '{header}' of {table.get('id')}"
        return found

    def _scalar_from_text(self, text: str, already: dict[str, str]) -> dict[str, str]:
        found: dict[str, str] = {}
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for field_name in self.schema.scalar_fields:
            if field_name in already:
                continue
            for keyword in sorted(self.schema.synonyms.get(field_name, []), key=len, reverse=True):
                if len(keyword) > 60:      # descriptive hints are meant for the LLM only
                    continue
                # Single-word keywords ("Date", "LTB") must start the line to avoid false hits.
                anchor = r"^" if len(keyword.split()) == 1 else r"(?<![A-Za-z])"
                pattern = re.compile(anchor + re.escape(keyword) + r"\s*(?:#|No\.?)?\s*[:\-]\s*(.+)", re.I)
                match = next(
                    (m for m in (pattern.search(ln) for ln in lines) if m and _valid_value(field_name, m.group(1))),
                    None,
                )
                if match:
                    found[field_name] = match.group(1).strip()[:120]
                    self.provenance["fields"][field_name] = f"text keyword '{keyword}'"
                    break
        if "Manufacturer_Name" in self.schema.scalar_fields and "Manufacturer_Name" not in already:
            m = re.search(
                r"\b([A-Z][A-Za-z0-9&.\-]*(?:[ \t]+[A-Z][A-Za-z0-9&.\-]*){0,4}),?[ \t]+"
                r"(Inc\.?|Incorporated|Corporation|Corp\.?|Ltd\.?|Limited|GmbH|AG|N\.V\.?|LLC|S\.A\.)(?![A-Za-z])",
                text,
            )
            if m:
                sep = ", " if m.group(2).lower().startswith("inc") else " "
                found["Manufacturer_Name"] = f"{m.group(1)}{sep}{m.group(2)}"
                self.provenance["fields"]["Manufacturer_Name"] = "company-suffix pattern"
        return found

    def _scalar_from_llm(self, text: str, kv_lines: list[str], missing: list[str]) -> dict[str, str]:
        if not self.llm or not missing:
            return {}
        hints = {f: self.schema.synonyms.get(f, []) for f in missing}
        context = ("\n".join(kv_lines) + "\n\n" + text)[: self.settings.llm_max_context_chars]
        prompt = (
            "Extract the following fields from this product change / discontinuance notice.\n"
            f"Fields and typical labels:\n{json.dumps(hints, indent=2)}\n\n"
            "Rules: copy values exactly as written; dates as YYYY-MM-DD; empty string if absent.\n\n"
            f"Document:\n\"\"\"\n{context}\n\"\"\""
        )
        schema = {"type": "object", "properties": {f: {"type": "string"} for f in missing}, "required": missing}
        result = self.llm.complete_json(prompt, schema) or {}
        found = {}
        for f in missing:
            value = result.get(f)
            if isinstance(value, list):
                value = value[0] if value else ""
            if isinstance(value, str) and _valid_value(f, value):
                found[f] = value.strip()
                self.provenance["fields"][f] = "llm"
        return found

    # ---------------- entry point ---------------- #
    def map(self, document: dict[str, Any]) -> dict[str, Any]:
        tables = document.get("tables", [])
        text = document.get("full_text", "")
        out = self.schema.empty_output()

        mapped = self._map_columns(tables)
        lists = self._list_tables(tables, mapped)
        self._fill_table_rows(out, tables, mapped, lists)

        scalars = self._scalar_from_tables(tables)
        scalars.update(self._scalar_from_text(text, scalars))
        missing = [f for f in self.schema.scalar_fields if f not in scalars]
        kv_lines = [f"{k} {v}" for t in tables if t["type"] == "key_value" for k, v in t["rows"]]
        scalars.update(self._scalar_from_llm(text, kv_lines, missing))

        for field_name, value in scalars.items():
            out[field_name] = normalize_date(value) if _is_date_field(field_name) else value
        return out


def map_to_pcn_schema(
    document: dict[str, Any],
    settings: Settings | None = None,
    llm: LLMClient | None = None,
    use_llm: bool = True,
) -> dict[str, Any]:
    """Map a SPARTAN document to the PCN schema and return the front-end compatible dict."""
    settings = settings or get_settings()
    if llm is None and use_llm:
        llm = create_llm_client(settings)
    mapper = PcnMapper(PcnSchema.load(), llm, settings)
    result = mapper.map(document)
    document["pcn_provenance"] = mapper.provenance
    return result
