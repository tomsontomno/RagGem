"""FastAPI server: HTTP surface for website integration.

Auth model - two tiers controlled by env vars:

  - API_ADMIN_KEY  -> full access to every endpoint, including upload,
                     delete, rebuild, destroy.
  - API_PUBLIC_KEY -> query-only. Safe to ship to a website client because
                     the worst it can do is ask questions.

The server refuses to start if API_ADMIN_KEY is unset, because exposing
admin endpoints without authentication would be unsafe by default.

All endpoints under ``/api/v1`` use a single header (``X-API-Key`` by
default; configurable via ``API_KEY_HEADER``).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Path as ApiPath,
    Security,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from .config import (
    API_ADMIN_KEY,
    API_KEY_HEADER,
    API_PUBLIC_KEY,
    APP_DESCRIPTION,
    APP_NAME,
    APP_VERSION,
    CORS_ORIGINS,
    MAX_QUESTION_CHARS,
    MAX_UPLOAD_BYTES,
    admin_key_required,
)
from .core.brains import (
    brain_paths,
    delete_file,
    destroy_brain,
    list_brains,
    list_files,
    reset_vector_store,
    save_uploaded_file,
)
from .core.engine import (
    RagEngineError,
    get_engine,
    invalidate_brain,
    query_brain,
    remove_source,
)
from .security import InvalidInput

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_CHARS)


class SourceMetadata(BaseModel):
    source_id: int
    title: str
    url: str
    source: str
    page: int
    content: str
    distance: float


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceMetadata]
    metadata: Dict[str, Any]


class UploadResponse(BaseModel):
    brain_id: str
    filename: str
    size_bytes: int
    chunks_indexed: int


class DeleteResponse(BaseModel):
    brain_id: str
    filename: str
    file_removed: bool
    chunks_removed: int


class FileInfoResponse(BaseModel):
    filename: str
    size_bytes: int
    extension: str


class BrainListResponse(BaseModel):
    brains: List[str]


class FilesResponse(BaseModel):
    brain_id: str
    files: List[FileInfoResponse]


class RebuildResponse(BaseModel):
    brain_id: str
    reindexed: Dict[str, int]
    total_chunks: int


class DestroyResponse(BaseModel):
    brain_id: str
    destroyed: bool


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
api_key_header = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)


def _require_admin(api_key: Optional[str] = Security(api_key_header)) -> None:
    """Require the admin API key. Used for any write endpoint."""
    if not API_ADMIN_KEY:
        # admin_key_required() in lifespan should have aborted startup -
        # this guard is defence in depth.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="server is not configured for authenticated requests",
        )
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"missing {API_KEY_HEADER} header",
        )
    if api_key != API_ADMIN_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="invalid credentials",
        )


def _require_query(api_key: Optional[str] = Security(api_key_header)) -> None:
    """Accept either admin or (if configured) public key."""
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"missing {API_KEY_HEADER} header",
        )
    if api_key == API_ADMIN_KEY:
        return
    if API_PUBLIC_KEY and api_key == API_PUBLIC_KEY:
        return
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="invalid credentials",
    )


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
@asynccontextmanager
async def _lifespan(app: FastAPI):  # noqa: ARG001 - FastAPI lifecycle hook
    admin_key_required()
    logger.info("%s v%s ready (CORS origins: %s)", APP_NAME, APP_VERSION, CORS_ORIGINS)
    yield
    logger.info("%s shutting down", APP_NAME)


app = FastAPI(
    title=APP_NAME,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------
def _map_invalid_input(exc: InvalidInput) -> HTTPException:
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


# ---------------------------------------------------------------------------
# System endpoints
# ---------------------------------------------------------------------------
@app.get("/health", tags=["System"])
async def health() -> Dict[str, Any]:
    """Liveness probe - no auth required."""
    return {
        "status": "ok",
        "name": APP_NAME,
        "version": APP_VERSION,
    }


# ---------------------------------------------------------------------------
# Brain CRUD
# ---------------------------------------------------------------------------
@app.get(
    "/api/v1/brains",
    response_model=BrainListResponse,
    tags=["Brain"],
    dependencies=[Depends(_require_admin)],
)
async def api_list_brains() -> BrainListResponse:
    return BrainListResponse(brains=list_brains())


@app.get(
    "/api/v1/brains/{brain_id}/files",
    response_model=FilesResponse,
    tags=["Brain"],
    dependencies=[Depends(_require_admin)],
)
async def api_list_files(brain_id: str = ApiPath(..., min_length=1)) -> FilesResponse:
    try:
        files = list_files(brain_id)
    except InvalidInput as exc:
        raise _map_invalid_input(exc) from exc
    return FilesResponse(
        brain_id=brain_id,
        files=[
            FileInfoResponse(
                filename=f.filename,
                size_bytes=f.size_bytes,
                extension=f.extension,
            )
            for f in files
        ],
    )


@app.post(
    "/api/v1/brains/{brain_id}/files",
    response_model=UploadResponse,
    tags=["Brain"],
    dependencies=[Depends(_require_admin)],
)
async def api_upload_file(
    brain_id: str = ApiPath(..., min_length=1),
    file: UploadFile = File(...),
) -> UploadResponse:
    """Upload a document to a brain and immediately index it."""
    if file.filename is None:
        raise HTTPException(status_code=400, detail="file has no filename")
    # Quick size guard before streaming (best effort; full check is enforced
    # by save_uploaded_file).
    declared = file.size  # may be None for streamed uploads
    if declared is not None and declared > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"file exceeds {MAX_UPLOAD_BYTES} bytes",
        )
    try:
        stored = save_uploaded_file(brain_id, file.filename, file.file)
        chunks = get_engine(brain_id).index_file(stored.name)
        invalidate_brain(brain_id)
    except InvalidInput as exc:
        raise _map_invalid_input(exc) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(exc)
        ) from exc
    except RagEngineError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return UploadResponse(
        brain_id=brain_id,
        filename=stored.name,
        size_bytes=stored.stat().st_size,
        chunks_indexed=chunks,
    )


@app.delete(
    "/api/v1/brains/{brain_id}/files/{filename}",
    response_model=DeleteResponse,
    tags=["Brain"],
    dependencies=[Depends(_require_admin)],
)
async def api_delete_file(
    brain_id: str = ApiPath(..., min_length=1),
    filename: str = ApiPath(..., min_length=1),
) -> DeleteResponse:
    try:
        chunks_removed = remove_source(brain_id, filename)
        file_removed = delete_file(brain_id, filename)
        invalidate_brain(brain_id)
    except InvalidInput as exc:
        raise _map_invalid_input(exc) from exc
    if not file_removed and chunks_removed == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{filename!r} not found in brain {brain_id!r}",
        )
    return DeleteResponse(
        brain_id=brain_id,
        filename=filename,
        file_removed=file_removed,
        chunks_removed=chunks_removed,
    )


@app.post(
    "/api/v1/brains/{brain_id}/rebuild",
    response_model=RebuildResponse,
    tags=["Brain"],
    dependencies=[Depends(_require_admin)],
)
async def api_rebuild(
    brain_id: str = ApiPath(..., min_length=1),
) -> RebuildResponse:
    try:
        paths = brain_paths(brain_id)
    except InvalidInput as exc:
        raise _map_invalid_input(exc) from exc
    if not paths.knowledge_dir.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"brain {brain_id!r} has no knowledge directory",
        )
    reset_vector_store(brain_id)
    invalidate_brain(brain_id)
    try:
        counts = get_engine(brain_id).index_all()
    except RagEngineError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return RebuildResponse(
        brain_id=brain_id,
        reindexed=counts,
        total_chunks=sum(counts.values()),
    )


@app.delete(
    "/api/v1/brains/{brain_id}",
    response_model=DestroyResponse,
    tags=["Brain"],
    dependencies=[Depends(_require_admin)],
)
async def api_destroy_brain(
    brain_id: str = ApiPath(..., min_length=1),
) -> DestroyResponse:
    try:
        paths = brain_paths(brain_id)
        existed = paths.exists()
        destroy_brain(brain_id)
        invalidate_brain(brain_id)
    except InvalidInput as exc:
        raise _map_invalid_input(exc) from exc
    return DestroyResponse(brain_id=brain_id, destroyed=existed)


# ---------------------------------------------------------------------------
# Query endpoint (the public surface a website embeds)
# ---------------------------------------------------------------------------
@app.post(
    "/api/v1/brains/{brain_id}/query",
    response_model=QueryResponse,
    tags=["Brain"],
    dependencies=[Depends(_require_query)],
)
async def api_query(
    request: QueryRequest,
    brain_id: str = ApiPath(..., min_length=1),
) -> QueryResponse:
    try:
        result = query_brain(brain_id, request.question)
    except InvalidInput as exc:
        raise _map_invalid_input(exc) from exc
    except RagEngineError as exc:
        # Rate-limit / provider unavailability becomes 503 so a website
        # can show a "try again later" message; everything else is 500.
        code = (
            status.HTTP_503_SERVICE_UNAVAILABLE
            if "rate-limited" in str(exc)
            else status.HTTP_500_INTERNAL_SERVER_ERROR
        )
        raise HTTPException(status_code=code, detail=str(exc)) from exc
    return QueryResponse(**result)
