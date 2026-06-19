"""Brain lifecycle: create, list, destroy, file CRUD.

A "brain" is a named knowledge base, isolated on disk under two parallel
directory trees:

    data/knowledge/{brain_id}/   ← raw source documents
    data/chroma_db/{brain_id}/   ← persistent vector store

This module owns the filesystem half of that layout (raw documents,
directory creation, listing, deletion). The vector-store half is owned by
:mod:`raggem.core.ingest`.

All paths returned by functions in this module are guaranteed by
``safe_join`` to stay inside the configured DATA_DIR.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, List

from ..config import (
    CHROMA_DB_DIR,
    KNOWLEDGE_DIR,
    MAX_UPLOAD_BYTES,
    SUPPORTED_EXTENSIONS,
)
from ..security import (
    InvalidInput,
    safe_join,
    sanitize_filename,
    validate_brain_id,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BrainPaths:
    """The two directories that together define a brain on disk."""

    brain_id: str
    knowledge_dir: Path
    chroma_dir: Path

    def exists(self) -> bool:
        """True if either directory exists and is non-empty."""
        return (self.knowledge_dir.exists() and any(self.knowledge_dir.iterdir())) or (
            self.chroma_dir.exists() and any(self.chroma_dir.iterdir())
        )


@dataclass(frozen=True)
class FileInfo:
    """Metadata about a single document inside a brain."""

    filename: str
    size_bytes: int
    extension: str


def brain_paths(brain_id: str) -> BrainPaths:
    """Resolve the two filesystem paths for a brain, after validating the id.

    Args:
        brain_id: candidate brain identifier.

    Returns:
        Paths object with both directories (which may or may not exist).

    Raises:
        InvalidInput: if the brain id is malformed.
    """
    brain_id = validate_brain_id(brain_id)
    return BrainPaths(
        brain_id=brain_id,
        knowledge_dir=safe_join(KNOWLEDGE_DIR, brain_id),
        chroma_dir=safe_join(CHROMA_DB_DIR, brain_id),
    )


def list_brains() -> List[str]:
    """Return all brain ids that have either a knowledge dir or a vector store.

    The result is sorted alphabetically. Brains with only an empty dir are
    still listed - they exist, they just have no content.
    """
    seen: set[str] = set()
    for root in (KNOWLEDGE_DIR, CHROMA_DB_DIR):
        if not root.exists():
            continue
        for entry in root.iterdir():
            if entry.is_dir():
                try:
                    seen.add(validate_brain_id(entry.name))
                except InvalidInput:
                    # Skip directories that don't look like brain ids.
                    logger.warning(
                        "Skipping malformed brain directory %r under %s",
                        entry.name,
                        root,
                    )
    return sorted(seen)


def list_files(brain_id: str) -> List[FileInfo]:
    """List the source documents stored in a brain.

    Args:
        brain_id: brain identifier.

    Returns:
        Sorted list of FileInfo records, one per supported document file.
        Hidden files and unsupported extensions are excluded.
    """
    paths = brain_paths(brain_id)
    if not paths.knowledge_dir.exists():
        return []
    results: List[FileInfo] = []
    for entry in sorted(paths.knowledge_dir.iterdir()):
        if not entry.is_file():
            continue
        if entry.name.startswith("."):
            continue
        suffix = entry.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            continue
        stat = entry.stat()
        results.append(
            FileInfo(filename=entry.name, size_bytes=stat.st_size, extension=suffix)
        )
    return results


def _ensure_knowledge_dir(brain_id: str) -> Path:
    paths = brain_paths(brain_id)
    paths.knowledge_dir.mkdir(parents=True, exist_ok=True)
    return paths.knowledge_dir


def save_uploaded_file(brain_id: str, filename: str, source: BinaryIO) -> Path:
    """Persist a streamed upload into the brain's knowledge directory.

    The file is first written to a ``.partial`` sibling and then renamed
    into place, so a half-completed upload never appears as a valid
    document. Size is enforced incrementally.

    Args:
        brain_id: brain id.
        filename: caller-supplied filename (will be sanitised here).
        source: binary stream yielding the file contents.

    Returns:
        Final path of the saved file.

    Raises:
        InvalidInput: on filename or path validation failures.
        ValueError: if the upload exceeds MAX_UPLOAD_BYTES.
    """
    knowledge_dir = _ensure_knowledge_dir(brain_id)
    safe_name = sanitize_filename(filename)
    target = safe_join(knowledge_dir, safe_name)
    partial = target.with_suffix(target.suffix + ".partial")

    written = 0
    try:
        with partial.open("wb") as out:
            while True:
                chunk = source.read(64 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise ValueError(
                        f"upload exceeds {MAX_UPLOAD_BYTES} bytes"
                    )
                out.write(chunk)
        partial.replace(target)
    except Exception:
        # Best-effort cleanup; don't mask the original exception.
        if partial.exists():
            try:
                partial.unlink()
            except OSError:
                logger.exception("Failed to clean up partial upload %s", partial)
        raise
    logger.info("Stored %d bytes at %s", written, target)
    return target


def delete_file(brain_id: str, filename: str) -> bool:
    """Remove a single document file from a brain.

    Args:
        brain_id: brain id.
        filename: filename inside the brain. Will be sanitised before use.

    Returns:
        True if a file was deleted, False if it did not exist.

    Raises:
        InvalidInput: if filename is unsafe.
    """
    paths = brain_paths(brain_id)
    safe_name = sanitize_filename(filename)
    target = safe_join(paths.knowledge_dir, safe_name)
    if not target.exists():
        return False
    target.unlink()
    logger.info("Deleted %s", target)
    return True


def destroy_brain(brain_id: str) -> None:
    """Erase a brain entirely: documents and vector store.

    Idempotent. Safe to call on a non-existent brain.

    Args:
        brain_id: brain id.

    Raises:
        InvalidInput: if brain id is malformed.
    """
    paths = brain_paths(brain_id)
    for directory in (paths.knowledge_dir, paths.chroma_dir):
        if directory.exists():
            shutil.rmtree(directory)
            logger.info("Destroyed %s", directory)


def reset_vector_store(brain_id: str) -> None:
    """Erase only the vector store, leaving source documents intact.

    Used before a rebuild so we re-embed from scratch.

    Args:
        brain_id: brain id.
    """
    paths = brain_paths(brain_id)
    if paths.chroma_dir.exists():
        shutil.rmtree(paths.chroma_dir)
        logger.info("Cleared vector store at %s", paths.chroma_dir)
