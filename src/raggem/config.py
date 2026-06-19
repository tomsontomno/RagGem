"""Central configuration for RagGem.

All knobs live here. Tunables read from environment variables; defaults are
chosen for production safety (low temperature, strict grounding threshold,
modest size limits). Frontend-dev maintainers should only ever need to edit
the .env file, not this module.

Invariants:
 - All paths returned by this module are absolute and inside BASE_DIR.
 - Numeric tunables are validated against sane ranges at import time; a
   misconfigured value raises ValueError before the server can start.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv

load_dotenv()


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name, default)
    return value.strip()


def _env_int(name: str, default: int, *, lo: int, hi: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"Env var {name!r} must be an integer, got {raw!r}") from exc
    if not lo <= value <= hi:
        raise ValueError(f"Env var {name!r}={value} outside [{lo}, {hi}]")
    return value


def _env_float(name: str, default: float, *, lo: float, hi: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"Env var {name!r} must be a number, got {raw!r}") from exc
    if not lo <= value <= hi:
        raise ValueError(f"Env var {name!r}={value} outside [{lo}, {hi}]")
    return value


def _env_list(name: str, default: str) -> List[str]:
    raw = os.environ.get(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# BASE_DIR points at the project root (the directory that contains src/).
BASE_DIR: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = BASE_DIR / "data"
KNOWLEDGE_DIR: Path = DATA_DIR / "knowledge"
CHROMA_DB_DIR: Path = DATA_DIR / "chroma_db"

KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# App metadata
# ---------------------------------------------------------------------------
APP_NAME: str = "RagGem"
APP_DESCRIPTION: str = (
    "Hardened RAG service: query a curated knowledge base from a website "
    "chatbot or from the terminal, with strict grounding and prompt-"
    "injection defences."
)
APP_VERSION: str = "2.0.0"


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
# Two-tier API keys.
#   - API_ADMIN_KEY: full access (upload, delete, rebuild, destroy, query).
#   - API_PUBLIC_KEY: read-only (only the /query endpoint). Safe to embed in
#     a website if combined with rate limiting / CORS allowlist.
#
# Backwards-compat: if only the legacy API_SECRET_KEY is set, it becomes the
# admin key and no public key exists.
API_ADMIN_KEY: str = _env_str(
    "API_ADMIN_KEY",
    _env_str("API_SECRET_KEY", ""),
)
API_PUBLIC_KEY: str = _env_str("API_PUBLIC_KEY", "")

# Header name used for API key auth.
API_KEY_HEADER: str = _env_str("API_KEY_HEADER", "X-API-Key")


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
# Comma-separated list of origins, or "*" for any (dev only).
CORS_ORIGINS: List[str] = _env_list("CORS_ORIGINS", "*")


# ---------------------------------------------------------------------------
# Model / generation
# ---------------------------------------------------------------------------
MODEL_NAME: str = _env_str("MODEL_NAME", "models/gemini-flash-lite-latest")
EMBEDDING_MODEL: str = _env_str("EMBEDDING_MODEL", "models/gemini-embedding-2")

# Embedding batch tuning. Each batch is ONE batchEmbedContents call.
#   EMBED_BATCH_SIZE  - records per call (free-tier-safe at 80).
#   EMBED_BATCH_GAP_S - pause between batches to respect the free-tier rate
#     limit. Set to 0 (paid key / small corpus) for fast ingestion; the demo
#     overrides this via env so live ingestion stays snappy.
EMBED_BATCH_SIZE: int = _env_int("EMBED_BATCH_SIZE", 80, lo=1, hi=250)
EMBED_BATCH_GAP_S: int = _env_int("EMBED_BATCH_GAP_S", 65, lo=0, hi=600)

# Temperature: low by default. 0.0 = greedy / fully deterministic per call.
TEMPERATURE: float = _env_float("TEMPERATURE", 0.0, lo=0.0, hi=1.0)

# Cap on response length (LLM-side, soft) and on system-side hard truncation.
MAX_OUTPUT_TOKENS: int = _env_int("MAX_OUTPUT_TOKENS", 1024, lo=64, hi=8192)
MAX_ANSWER_CHARS: int = _env_int("MAX_ANSWER_CHARS", 2000, lo=200, hi=10000)

# Thinking-token budget for Gemini 2.5 / 3.x models. RAG over a fixed
# corpus does not benefit from chain-of-thought reasoning - every fact the
# bot needs is already in the retrieved context - so the default disables
# it for predictable, low-latency answers.
#   0  -> disable thinking (recommended for RAG)
#   N  -> allow up to N thinking tokens
#  -1  -> dynamic, let the model decide
THINKING_BUDGET: int = _env_int("THINKING_BUDGET", 0, lo=-1, hi=32768)


# ---------------------------------------------------------------------------
# Retrieval / chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE: int = _env_int("CHUNK_SIZE", 900, lo=200, hi=4000)
CHUNK_OVERLAP: int = _env_int("CHUNK_OVERLAP", 120, lo=0, hi=1000)
RETRIEVAL_K: int = _env_int("RETRIEVAL_K", 6, lo=1, hi=20)

# Distance threshold for "I don't know". ChromaDB returns cosine *distance*
# (0 = identical, 2 = opposite). Anything beyond this for the best match
# triggers a refusal before we even call the LLM. Tunable per deployment.
GROUNDING_MAX_DISTANCE: float = _env_float(
    "GROUNDING_MAX_DISTANCE",
    0.95,
    lo=0.0,
    hi=2.0,
)


# ---------------------------------------------------------------------------
# Input limits
# ---------------------------------------------------------------------------
# Maximum size of an uploaded file (bytes). 25 MiB default.
MAX_UPLOAD_BYTES: int = _env_int(
    "MAX_UPLOAD_BYTES",
    25 * 1024 * 1024,
    lo=1024,
    hi=500 * 1024 * 1024,
)

# Maximum characters in a user question. Anything longer is rejected.
MAX_QUESTION_CHARS: int = _env_int("MAX_QUESTION_CHARS", 2000, lo=16, hi=10000)

# Maximum filename length (after sanitisation).
MAX_FILENAME_CHARS: int = _env_int("MAX_FILENAME_CHARS", 200, lo=8, hi=255)

# Supported document extensions (lower-case, including the dot).
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".txt", ".md", ".markdown", ".json"})


# ---------------------------------------------------------------------------
# Site identity (used inside the system prompt)
# ---------------------------------------------------------------------------
# Optional. When set, the assistant identifies itself as the assistant for
# this site. Keep it short.
SITE_NAME: str = _env_str("SITE_NAME", "this site")


# ---------------------------------------------------------------------------
# Public state
# ---------------------------------------------------------------------------
def admin_key_required() -> str:
    """Return the admin API key, or raise if unset.

    Used by the API layer to ensure the service refuses to start with no
    admin credentials configured - silent default keys are a footgun.
    """
    if not API_ADMIN_KEY:
        raise RuntimeError(
            "API_ADMIN_KEY (or legacy API_SECRET_KEY) is not configured. "
            "Refusing to expose admin endpoints with no authentication."
        )
    return API_ADMIN_KEY
