"""Configuration for the SPARTAN pipeline.

Two kinds of configuration live here:

* **Heuristic parameters** (``*Params`` dataclasses). Every pixel threshold is
  expressed for a page rendered at :data:`REFERENCE_DPI` (300 DPI, the value used
  in the paper) and is scaled automatically when another DPI is used, so the
  heuristics stay resolution-independent.
* **Runtime settings** (:class:`Settings`). Read from environment variables with
  the ``SPARTAN_`` prefix or from a ``.env`` file. Secrets (API keys) must only
  ever come from here, never from source code.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REFERENCE_DPI = 300
PACKAGE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = PACKAGE_DIR / "assets"


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #
class TextSourceMode(str, Enum):
    """Where cell text comes from."""

    AUTO = "auto"   # PDF text layer, OCR only for pages/areas without one
    PDF = "pdf"     # PDF text layer only
    OCR = "ocr"     # always OCR (reproduces the paper's original setup)


class OcrScope(str, Enum):
    """How OCR is applied to table cells (paper: Single-cell / Multi-cell / auto)."""

    PAGE = "page"       # OCR the page once, assign words to cells geometrically
    SINGLE = "single"   # one OCR call per cell
    MULTI = "multi"     # chunk cells of a column with the row-separator image
    AUTO = "auto"       # single if the table has < auto_threshold cells, else multi


class LlmProvider(str, Enum):
    NONE = "none"
    OLLAMA = "ollama"
    AZURE = "azure"


# --------------------------------------------------------------------------- #
# Heuristic parameters (values at 300 DPI)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PreprocessParams:
    adaptive_block_size: int = 11
    adaptive_c: int = 2
    ink_threshold: int = 200            # gray value below which a pixel counts as ink
    canny_low: int = 50
    canny_high: int = 150


@dataclass(frozen=True)
class ColumnParams:
    min_gap_height: int = 200           # paper: gaps shorter than 200 px are ignored
    min_gap_height_ratio: float = 0.30  # ...and shorter than 30% of the content height
    min_gap_width: int = 20             # a gutter must be at least this wide
    text_proximity: int = 15            # paper: text within 15 px on both sides of the gap
    min_side_ink_ratio: float = 0.02    # fraction of the gap height that must touch text
    suppression_distance: int = 300     # a longer gap within this distance wins
    merge_distance: int = 15            # paper: gaps with x closer than 15 px are merged
    segment_margin: int = 5


@dataclass(frozen=True)
class RegionParams:
    # Cropping outer boundaries
    crop_min_area_ratio: float = 0.75   # paper: contour covering >= 75% of the page
    crop_margin: int = 25               # paper: 25 px safety margin
    # Boundary-based detector (paper Table 3)
    line_kernel: int = 25
    min_cell_width: int = 100
    min_cell_height: int = 20
    cell_gap_threshold: int = 20
    min_table_width: int = 500
    min_table_height: int = 100
    # Text-based (borderless) detector
    align_tolerance: int = 5            # paper: 5 px for left / centre alignment
    phrase_gap_ratio: float = 0.8       # words closer than this x line-height form one phrase
    min_column_gap: int = 40            # whitespace separating two table cells on a line
    min_rows: int = 3                   # minimum number of lines in a borderless table
    min_columns: int = 2
    min_text_table_width: int = 300
    use_right_alignment: bool = True    # extension: right-aligned (numeric) columns
    pseudo_line_thickness: int = 2
    pseudo_padding: int = 8


@dataclass(frozen=True)
class ParserParams:
    min_vertical_segment: int = 33
    min_horizontal_segment: int = 60
    max_skew: float = 2.0               # max |dx| (vertical) / |dy| (horizontal) in px
    vertical_cluster_tolerance: int = 10
    horizontal_cluster_tolerance: int = 6
    gap_tolerance: int = 20             # collinear pieces closer than this are joined
    min_vertical_line: int = 40
    min_horizontal_line: int = 80
    endpoint_tolerance: int = 15
    edge_coverage_ratio: float = 0.5    # a cell edge exists if lines cover >= 50% of it
    min_cell_size: int = 15
    erase_text: bool = True             # mask words before line detection


@dataclass(frozen=True)
class CurationParams:
    kv_max_rows: int = 5                # paper: key-value tables have <= 5 rows
    kv_width_ratio: float = 1.5         # paper: "significant" column width difference
    kv_colon_ratio: float = 0.6         # extension: most keys end with ':'
    max_header_rows: int = 3
    cell_shrink: int = 3                # paper: shrink cells by 3 px for OCR
    context_title_max_chars: int = 150
    merge_width_tolerance: int = 15
    merge_width_tolerance_ratio: float = 0.03
    header_similarity: float = 0.85


@dataclass(frozen=True)
class OcrParams:
    scope: OcrScope = OcrScope.PAGE
    auto_threshold: int = 120           # paper: single-cell below 120 cells
    chunk_size: int = 15                # paper: 10, 15 or 20 (15 used in the evaluation)
    row_separator_text: str = "RSRSRSRS@"
    row_separator_image: Path = ASSETS_DIR / "row-separator.png"
    min_text_layer_chars: int = 10      # fewer chars than this -> page treated as scanned
    unexplained_ink_ratio: float = 0.03 # ink not covered by text/lines that triggers hybrid OCR
    tesseract_config: str = "--oem 3 --psm 6"
    page_tesseract_config: str = "--oem 3 --psm 11"


# --------------------------------------------------------------------------- #
# Pipeline configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PipelineConfig:
    """Everything a single pipeline run needs."""

    dpi: int = REFERENCE_DPI
    # Pipeline steps (paper Table 2 CLI flags)
    preprocess: bool = True
    column_detection: bool = False
    region_segmentation: bool = True
    edge_detection: bool = False
    # Text extraction
    text_source: TextSourceMode = TextSourceMode.AUTO
    ocr_engine: str = "tesseract"        # "tesseract", "azure", "none" or "package.module:Class"
    # Outputs
    save_intermediate: bool = False
    output_dir: Path = Path("output")
    export_csv: bool = False
    # Optional PCN schema-mapping plugin
    pcn_mapping: bool = False
    # Heuristics
    preprocess_params: PreprocessParams = field(default_factory=PreprocessParams)
    column_params: ColumnParams = field(default_factory=ColumnParams)
    region_params: RegionParams = field(default_factory=RegionParams)
    parser_params: ParserParams = field(default_factory=ParserParams)
    curation_params: CurationParams = field(default_factory=CurationParams)
    ocr_params: OcrParams = field(default_factory=OcrParams)

    @property
    def scale(self) -> float:
        return self.dpi / REFERENCE_DPI

    def px(self, value: float) -> int:
        """Scale a 300-DPI pixel threshold to the configured DPI."""
        return max(1, int(round(value * self.scale)))

    def with_overrides(self, **changes) -> "PipelineConfig":
        return replace(self, **changes)


# --------------------------------------------------------------------------- #
# Runtime settings (environment / .env)
# --------------------------------------------------------------------------- #
class Settings(BaseSettings):
    """Environment-driven settings. See ``.env.example`` for all variables."""

    model_config = SettingsConfigDict(env_prefix="SPARTAN_", env_file=".env", extra="ignore")

    # OCR
    tesseract_cmd: str | None = None
    azure_vision_endpoint: str | None = None
    azure_vision_key: str | None = None

    # LLM plugin
    llm_provider: LlmProvider = LlmProvider.OLLAMA
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "gemma4:latest"
    llm_timeout: float = 180.0
    llm_max_context_chars: int = 12000
    azure_openai_endpoint: str | None = None
    azure_openai_key: str | None = None
    azure_openai_deployment: str = "gpt-4o"
    azure_openai_api_version: str = "2024-06-01"

    # API
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    max_upload_mb: int = 50


_settings: Settings | None = None


def get_settings() -> Settings:
    """Process-wide settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
