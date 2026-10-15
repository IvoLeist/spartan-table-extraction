"""REST API (FastAPI).

Run with::

    uvicorn spartan.api:app --host 127.0.0.1 --port 8000
    # or
    python -m spartan.api

Endpoints:

* ``GET  /health``      - liveness and configuration summary.
* ``POST /extract``     - full SPARTAN document (tables, text, pages).
* ``POST /upload-pdf``  - PCN-schema output used by the SPARTAN front-end.

The pipeline is CPU-bound, so endpoints are plain (sync) functions that FastAPI
runs in its thread pool; the event loop is never blocked.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

try:
    from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError('The REST API needs the "api" extra: pip install "spartan-tables[api]"') from exc

from . import __version__
from .config import PipelineConfig, get_settings
from .export import public_document
from .pipeline import SpartanPipeline

log = logging.getLogger(__name__)
settings = get_settings()

app = FastAPI(
    title="SPARTAN",
    version=__version__,
    description="Structured Parsing and Relevant Table Analysis - table extraction from PDF documents.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_pipelines: dict[bool, SpartanPipeline] = {}


def _pipeline(column_detection: bool) -> SpartanPipeline:
    if column_detection not in _pipelines:
        _pipelines[column_detection] = SpartanPipeline(PipelineConfig(column_detection=column_detection))
    return _pipelines[column_detection]


@app.exception_handler(HTTPException)
async def _http_error(_: Request, exc: HTTPException) -> JSONResponse:
    # The front-end reads `error` from failed responses.
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


def _save_upload(file: UploadFile) -> Path:
    limit = settings.max_upload_mb * 1024 * 1024
    data = file.file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(status_code=413, detail=f"File larger than {settings.max_upload_mb} MB")
    if not data.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    fd, name = tempfile.mkstemp(suffix=".pdf")
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)
    return Path(name)


def _run(file: UploadFile, column_detection: bool, pcn: bool) -> dict[str, Any]:
    path = _save_upload(file)
    try:
        name = Path(file.filename or "document.pdf").stem
        document = _pipeline(column_detection).process(path, document_name=name)
        if pcn:
            from .plugins.pcn import map_to_pcn_schema

            document["pcn"] = map_to_pcn_schema(document)
        return public_document(document)
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Processing failed")
        raise HTTPException(status_code=500, detail=f"Processing failed: {exc}") from exc
    finally:
        path.unlink(missing_ok=True)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "version": __version__, "llm_provider": settings.llm_provider.value}


@app.post("/extract")
def extract(
    file: UploadFile = File(...),
    column_detection: bool = Query(False, description="enable column detection"),
    pcn: bool = Query(False, description="also map to the PCN schema"),
) -> dict[str, Any]:
    """Return the full SPARTAN document."""
    return _run(file, column_detection, pcn)


@app.post("/upload-pdf")
def upload_pdf(file: UploadFile = File(...)) -> dict[str, Any]:
    """Return the PCN-schema view expected by the SPARTAN front-end."""
    return _run(file, column_detection=False, pcn=True)["pcn"]


def run() -> None:  # pragma: no cover - console entry point
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    uvicorn.run("spartan.api:app", host=os.getenv("SPARTAN_HOST", "127.0.0.1"), port=int(os.getenv("SPARTAN_PORT", "8000")))


if __name__ == "__main__":  # pragma: no cover
    run()
