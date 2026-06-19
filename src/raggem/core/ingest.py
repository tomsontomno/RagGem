"""Document ingestion: file -> chunks -> chromadb.

Pipeline:
  1. load_file        reads a file into Document objects
  2. chunk_documents  splits Documents into overlapping chunks
  3. embed_and_store  embeds chunks via google.genai and writes to chromadb

No LangChain. Uses fitz (PyMuPDF), google.genai, and chromadb directly.

Deterministic chunk IDs - ``{source}::{page}::{chunk_index}::{content_hash}`` -
make per-file deletion safe: remove by ``where={"source": filename}``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import chromadb
import fitz  # PyMuPDF
import google.genai as genai
import google.genai.types as genai_types

from ..config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBED_BATCH_GAP_S,
    EMBED_BATCH_SIZE,
    EMBEDDING_MODEL,
    SUPPORTED_EXTENSIONS,
)

logger = logging.getLogger(__name__)

_EMBED_MAX_RETRIES = 4
_EMBED_BASE_DELAY_S = 15
# Batch size + inter-batch gap are env-tunable (see config). Each batch is one
# batchEmbedContents call; the gap guards the free-tier rate limit and can be
# set to 0 with a paid key or a small corpus for fast ingestion.
_EMBED_BATCH_SIZE = EMBED_BATCH_SIZE
_EMBED_BATCH_GAP_S = EMBED_BATCH_GAP_S

_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""]


# ---------------------------------------------------------------------------
# Document dataclass
# ---------------------------------------------------------------------------
@dataclass
class Document:
    """A text fragment extracted from a source file.

    Invariants:
      - ``page_content`` is a non-empty string.
      - ``metadata["source"]`` is always the filename (no path components).
      - ``metadata["page"]`` is a non-negative int.
      - ``metadata["chunk"]`` is set by ``chunk_documents``.
    """

    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# JSON record helpers
# ---------------------------------------------------------------------------
_ARRAY_KEYS = ("records", "items", "data", "results", "entries", "pieces", "albums")
# ChromaDB requires metadata values to be str | int | float | bool.
_META_MAX_CHARS = 500


def _flatten_value(val: Any) -> str | int | float | bool | None:
    """Convert a JSON value to a ChromaDB-compatible scalar.

    Precondition:  ``val`` is any JSON-decoded Python value.
    Postcondition: returns str, int, float, bool, or None (None -> field is
                   dropped from metadata); strings are truncated at
                   _META_MAX_CHARS; lists are joined with ", ".
    """
    if val is None or val == "" or val == []:
        return None
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return val
    if isinstance(val, str):
        return val[:_META_MAX_CHARS]
    if isinstance(val, list):
        joined = ", ".join(str(v) for v in val if v is not None)
        return joined[:_META_MAX_CHARS] if joined else None
    if isinstance(val, dict):
        serialised = "; ".join(f"{k}={v}" for k, v in val.items() if v is not None)
        return serialised[:_META_MAX_CHARS] if serialised else None
    return str(val)[:_META_MAX_CHARS]


def _record_to_document(source: str, idx: int, record: Any) -> Document:
    """Convert a single JSON record to a Document ready for embedding.

    Each field is serialised to ``"key: value"`` and joined with ``". "``
    to form the ``page_content`` that gets embedded.  Scalar and list
    metadata fields are stored separately so the engine can surface them
    as citation fields (title, album, etc.).

    Precondition:
        ``source`` is a non-empty filename string.
        ``idx`` is a non-negative int (record's position in the array).
        ``record`` is any JSON-decoded Python value.
    Postcondition:
        Returns a Document whose ``page_content`` is non-empty.
        ``metadata`` always contains "source" (str), "page" (int), and
        "record_index" (int); all other metadata values are ChromaDB-
        compatible (str | int | float | bool).
    """
    base_meta: Dict[str, Any] = {
        "source": source,
        "page": idx,
        "record_index": idx,
    }

    if not isinstance(record, dict):
        return Document(
            page_content=str(record) or f"record {idx}",
            metadata=base_meta,
        )

    parts: List[str] = []
    extra_meta: Dict[str, Any] = {}

    for key, val in record.items():
        flat = _flatten_value(val)
        if flat is None:
            continue
        parts.append(f"{key}: {flat}")
        extra_meta[key] = flat

    base_meta.update(extra_meta)
    # Re-assert ingest-authoritative fields so a record field named "source",
    # "page", or "record_index" cannot shadow the values set by this loader.
    base_meta["source"] = source
    base_meta["page"] = idx
    base_meta["record_index"] = idx
    page_content = ". ".join(parts) if parts else f"record {idx}"
    return Document(page_content=page_content, metadata=base_meta)


def _load_json_records(path: Path) -> List[Document]:
    """Parse a JSON file and return one Document per logical record.

    Auto-detection rules (in order):
      1. JSON array        -> one Document per element.
      2. JSON object with a known array key (``records``, ``items``, ...)
                           -> one Document per element of that array.
      3. Any other value   -> one Document wrapping the whole value.

    Precondition:
        ``path`` exists and is readable; ``path.suffix.lower() == ".json"``.
    Postcondition:
        Returns a list of Documents (may be empty for an empty array).
        Raises ``ValueError`` with context on malformed JSON.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path.name!r}: {exc}") from exc

    source = path.name

    if isinstance(raw, list):
        return [_record_to_document(source, i, item) for i, item in enumerate(raw)]

    if isinstance(raw, dict):
        for key in _ARRAY_KEYS:
            if isinstance(raw.get(key), list):
                return [
                    _record_to_document(source, i, item)
                    for i, item in enumerate(raw[key])
                ]
        # Single top-level object -> one record.
        return [_record_to_document(source, 0, raw)]

    # Scalar (str, int, bool, null) -> one document.
    return [_record_to_document(source, 0, raw)]


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------
def load_file(path: Path) -> List[Document]:
    """Load a supported file into Document objects.

    Args:
        path: absolute path to the file; extension must be in
            SUPPORTED_EXTENSIONS.

    Returns:
        For PDFs: one Document per page (only non-blank pages).
        For text/markdown: a single Document with the full file content.

    Raises:
        ValueError: unsupported extension.
        FileNotFoundError: path does not exist (raised by fitz / read_text).
    """
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported extension {suffix!r} for {path.name}")

    if suffix == ".pdf":
        docs: List[Document] = []
        pdf = fitz.open(str(path))
        try:
            for page_num, page in enumerate(pdf):
                text = page.get_text()
                if text.strip():
                    docs.append(Document(
                        page_content=text,
                        metadata={"source": path.name, "page": page_num},
                    ))
        finally:
            pdf.close()
        return docs

    if suffix == ".json":
        return _load_json_records(path)

    # Plain text / markdown
    text = path.read_text(encoding="utf-8", errors="replace")
    return [Document(page_content=text, metadata={"source": path.name, "page": 0})]


# ---------------------------------------------------------------------------
# Text splitting
# ---------------------------------------------------------------------------
def _split_on_sep(text: str, sep: str) -> List[str]:
    """Split ``text`` on ``sep``, keeping the separator attached to each piece.

    Example: ``_split_on_sep("a\\n\\nb\\n\\nc", "\\n\\n")``
    -> ``["a\\n\\n", "b\\n\\n", "c"]``

    Postcondition: ``"".join(result)`` reconstructs the original ``text``
    (minus the leading empty strings that are discarded).
    """
    if not sep:
        return list(text)
    parts = text.split(sep)
    result: List[str] = []
    for i, part in enumerate(parts):
        if not part:
            continue
        result.append(part + sep if i < len(parts) - 1 else part)
    return result


def _merge_chunks(pieces: List[str], size: int, overlap: int) -> List[str]:
    """Greedily merge ``pieces`` into chunks of at most ``size`` chars.

    When a flush happens, the tail of the current buffer (up to ``overlap``
    chars) is carried into the next chunk.

    Postcondition: every element of the result has len ≤ max(size, max(len(p)))
    and result is non-empty if pieces is non-empty.
    """
    chunks: List[str] = []
    buf: List[str] = []
    buf_len = 0

    for piece in pieces:
        piece_len = len(piece)
        if buf_len + piece_len > size and buf:
            chunks.append("".join(buf))
            # Overlap: keep tail pieces whose total ≤ overlap chars.
            tail: List[str] = []
            tail_len = 0
            for p in reversed(buf):
                if tail_len + len(p) <= overlap:
                    tail.insert(0, p)
                    tail_len += len(p)
                else:
                    break
            buf = tail
            buf_len = tail_len
        buf.append(piece)
        buf_len += piece_len

    if buf:
        chunks.append("".join(buf))

    return [c.strip() for c in chunks if c.strip()]


def _recursive_chunk(text: str, size: int, overlap: int, seps: List[str]) -> List[str]:
    """Recursively split ``text`` into chunks of at most ``size`` chars.

    Tries separators in order; falls back to character-level slicing if none
    produce multiple pieces.

    Postcondition: every element is non-empty (stripped); len ≤ size unless a
    single piece exceeds size with no more separators available.
    """
    if len(text) <= size:
        return [text] if text.strip() else []

    for i, sep in enumerate(seps):
        if not sep:
            # Character-level fallback (last separator in the list).
            step = max(1, size - overlap)
            return [
                text[j : j + size]
                for j in range(0, len(text), step)
                if text[j : j + size].strip()
            ]

        pieces = _split_on_sep(text, sep)
        if len(pieces) <= 1:
            continue  # separator not in text; try the next one

        remaining = seps[i + 1 :]
        merged = _merge_chunks(pieces, size, overlap)
        result: List[str] = []
        for chunk in merged:
            if len(chunk) > size and remaining:
                result.extend(_recursive_chunk(chunk, size, overlap, remaining))
            elif chunk.strip():
                result.append(chunk)
        return result

    # No separator matched at all - character slice.
    step = max(1, size - overlap)
    return [text[j : j + size] for j in range(0, len(text), step) if text[j : j + size].strip()]


def chunk_documents(documents: Iterable[Document]) -> List[Document]:
    """Split documents into overlapping chunks suitable for embedding.

    Args:
        documents: typically the output of :func:`load_file`.

    Returns:
        Flat list of chunked Documents. Each chunk carries the original
        ``source`` / ``page`` metadata and gains a ``chunk`` index.
    """
    chunks: List[Document] = []
    for doc in documents:
        src = str(doc.metadata.get("source", "unknown"))
        page = int(doc.metadata.get("page", 0))
        texts = _recursive_chunk(doc.page_content, CHUNK_SIZE, CHUNK_OVERLAP, _SEPARATORS)
        for idx, text in enumerate(texts):
            chunks.append(Document(
                page_content=text,
                metadata={**doc.metadata, "source": src, "page": page, "chunk": idx},
            ))
    return chunks


# ---------------------------------------------------------------------------
# ID generation
# ---------------------------------------------------------------------------
def _stable_id(source: str, page: int, idx: int, content: str) -> str:
    """Build a deterministic chunk ID from its location and content hash."""
    h = hashlib.sha256(content.encode("utf-8", errors="ignore")).hexdigest()[:16]
    return f"{source}::{page}::{idx}::{h}"


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------
def _embed_batch(client: genai.Client, texts: List[str]) -> List[List[float]]:
    """Embed a list of texts via google.genai.

    Precondition:  ``len(texts) > 0``; ``client`` carries a valid API key.
    Postcondition: ``len(result) == len(texts)``; each element is a float list.

    Wraps each text in a ``Content`` object. Passing raw strings as a list
    causes some embedding models (e.g. gemini-embedding-2) to treat the
    whole list as one multi-part document, returning a single embedding
    instead of one per text.
    """
    contents = [
        genai_types.Content(parts=[genai_types.Part(text=t)]) for t in texts
    ]
    response = client.models.embed_content(model=EMBEDDING_MODEL, contents=contents)
    return [list(e.values) for e in response.embeddings]


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
def embed_and_store(
    collection: chromadb.Collection,
    chunks: Sequence[Document],
    client: genai.Client,
) -> List[str]:
    """Embed chunks and persist them in a chromadb collection.

    Precondition:
        ``collection`` is an open chromadb Collection;
        ``chunks`` is non-empty;
        ``client`` has a valid GOOGLE_API_KEY.

    Postcondition:
        All chunks are stored; returns IDs with ``len(result) == len(chunks)``.
        Rate-limit errors trigger exponential-backoff retries before raising.
    """
    if not chunks:
        return []

    doc_list = list(chunks)
    ids = [
        _stable_id(
            source=str(c.metadata.get("source", "unknown")),
            page=int(c.metadata.get("page", 0)),
            idx=int(c.metadata.get("chunk", 0)),
            content=c.page_content,
        )
        for c in doc_list
    ]

    all_ids: List[str] = []

    for batch_start in range(0, len(doc_list), _EMBED_BATCH_SIZE):
        batch_docs = doc_list[batch_start : batch_start + _EMBED_BATCH_SIZE]
        batch_ids = ids[batch_start : batch_start + _EMBED_BATCH_SIZE]
        texts = [d.page_content for d in batch_docs]
        metadatas = [d.metadata for d in batch_docs]

        for attempt in range(_EMBED_MAX_RETRIES):
            try:
                embeddings = _embed_batch(client, texts)
                collection.add(
                    documents=texts,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=batch_ids,
                )
                all_ids.extend(batch_ids)
                logger.info(
                    "Embedded batch [%d:%d] (%d total)",
                    batch_start,
                    batch_start + len(batch_docs),
                    len(all_ids),
                )
                break
            except Exception as exc:
                msg = str(exc)
                is_rate_limited = any(
                    needle in msg.lower()
                    for needle in ("429", "resource_exhausted", "quota", "rate")
                )
                if not is_rate_limited or attempt == _EMBED_MAX_RETRIES - 1:
                    raise
                delay = _EMBED_BASE_DELAY_S * (2 ** attempt)
                logger.warning(
                    "Embedding rate-limited (attempt %d/%d); retrying in %ds. %s",
                    attempt + 1,
                    _EMBED_MAX_RETRIES,
                    delay,
                    msg.splitlines()[0][:120],
                )
                time.sleep(delay)

        if batch_start + _EMBED_BATCH_SIZE < len(doc_list):
            logger.info("Pausing %ds between embedding batches", _EMBED_BATCH_GAP_S)
            time.sleep(_EMBED_BATCH_GAP_S)

    return all_ids


def delete_source_from_store(collection: chromadb.Collection, source: str) -> int:
    """Remove every chunk whose ``source`` metadata equals ``source``.

    Args:
        collection: open chromadb Collection.
        source: filename (no path components).

    Returns:
        Number of chunks removed.
    """
    existing = collection.get(where={"source": source}, include=[])
    ids: List[str] = existing.get("ids") or []
    if ids:
        collection.delete(ids=ids)
        logger.info("Removed %d chunks for source %s", len(ids), source)
    return len(ids)
