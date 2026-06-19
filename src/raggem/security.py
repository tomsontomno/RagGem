"""Hardened input validation for RagGem.

Every entry point - CLI, HTTP, MCP - runs user-supplied strings through this
module before anything touches the filesystem, the vector store or the LLM.

Design notes:
 - Validators never *fix* invalid input. They raise InvalidInput so the
   caller can decide how to surface the error. Silent normalisation hides
   bugs.
 - Path containment is asserted with Path.resolve() + is_relative_to(),
   which is the only way to defeat ``..`` segments and absolute-path
   override attempts on POSIX.
 - The brain-id and filename regexes are intentionally tight - only
   characters that are filesystem-safe on Linux, macOS and Windows.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from .config import (
    MAX_FILENAME_CHARS,
    MAX_QUESTION_CHARS,
    SUPPORTED_EXTENSIONS,
)


class InvalidInput(ValueError):
    """Raised when user-supplied input fails validation."""


# Brain IDs are part of URL paths and filesystem paths. Keep them
# unambiguous: lowercase letters, digits, dash, underscore. Must start with
# an alphanumeric character.
_BRAIN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

# Filenames may include letters, digits, dot, dash, underscore, space.
# Must not start with a dot (no hidden files) or contain path separators.
_FILENAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_. \-]*$")


def validate_brain_id(brain_id: str) -> str:
    """Validate a brain identifier.

    Args:
        brain_id: candidate identifier from user input.

    Returns:
        The same string, unchanged, once validated.

    Raises:
        InvalidInput: if the identifier contains anything outside the safe
            alphabet or exceeds 64 characters.
    """
    if not isinstance(brain_id, str):
        raise InvalidInput("brain_id must be a string")
    if not _BRAIN_ID_RE.fullmatch(brain_id):
        raise InvalidInput(
            "brain_id must match ^[a-z0-9][a-z0-9_-]{0,63}$ "
            "(lowercase letters, digits, dash, underscore; 1-64 chars; "
            "must start with a letter or digit)."
        )
    return brain_id


def sanitize_filename(name: str) -> str:
    """Reduce a raw filename to a safe, portable form.

    Strips any directory components, normalises Unicode (NFKC), removes
    control characters, collapses whitespace, and asserts the result
    matches the safe-filename pattern.

    Args:
        name: candidate filename, e.g. from a multipart upload.

    Returns:
        A cleaned filename with extension preserved.

    Raises:
        InvalidInput: if the cleaned name is empty, too long, lacks a
            supported extension, or contains forbidden characters.
    """
    if not isinstance(name, str) or not name:
        raise InvalidInput("filename must be a non-empty string")

    # Reject control characters outright before any normalisation. Null
    # bytes, in particular, are never legitimate in a filename and are a
    # known attack signature on older file-handling code paths.
    if any((not ch.isprintable() or ord(ch) < 0x20) for ch in name):
        raise InvalidInput("filename contains control characters")

    # Drop any directory component (defeats `..\..\foo.pdf` style inputs).
    name = Path(name).name

    # Normalise Unicode so visually identical strings have a single
    # representation.
    name = unicodedata.normalize("NFKC", name)
    name = name.strip()

    # Collapse runs of whitespace.
    name = re.sub(r"\s+", " ", name)

    if not name or name in {".", ".."}:
        raise InvalidInput("filename is empty after sanitisation")
    if len(name) > MAX_FILENAME_CHARS:
        raise InvalidInput(
            f"filename exceeds {MAX_FILENAME_CHARS} characters after sanitisation"
        )
    if not _FILENAME_RE.fullmatch(name):
        raise InvalidInput(
            "filename contains forbidden characters; allowed: A-Z a-z 0-9 . _ - space"
        )

    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise InvalidInput(
            f"unsupported file extension {suffix!r}. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    return name


def safe_join(root: Path, *parts: str) -> Path:
    """Join path components and assert the result stays inside ``root``.

    This is the *only* sanctioned way to build a filesystem path from
    user-supplied components. It defeats ``..`` traversal, absolute-path
    overrides and symlink escapes (because resolve() follows symlinks).

    Args:
        root: trusted base directory; must already exist or be createable.
        *parts: untrusted path components.

    Returns:
        Resolved absolute path under ``root``.

    Raises:
        InvalidInput: if the resolved path falls outside ``root``.
    """
    root_resolved = root.resolve()
    candidate = root_resolved.joinpath(*parts).resolve()
    if root_resolved != candidate and root_resolved not in candidate.parents:
        raise InvalidInput("path escapes its allowed root")
    return candidate


def sanitize_question(question: str) -> str:
    """Validate and lightly clean a user question.

    We strip control characters and surrounding whitespace, normalise
    Unicode, and assert length bounds. We do *not* attempt to scrub
    "jailbreak phrases" here - the LLM-side defences in :mod:`prompts`
    handle adversarial content. Pre-filtering keywords is brittle and gives
    a false sense of security.

    Args:
        question: raw user question.

    Returns:
        Cleaned question string ready to be embedded into the prompt.

    Raises:
        InvalidInput: if empty after cleaning or too long.
    """
    if not isinstance(question, str):
        raise InvalidInput("question must be a string")
    cleaned = unicodedata.normalize("NFKC", question)
    cleaned = "".join(ch for ch in cleaned if ch.isprintable() or ch in "\n\t")
    cleaned = cleaned.strip()
    if not cleaned:
        raise InvalidInput("question is empty")
    if len(cleaned) > MAX_QUESTION_CHARS:
        raise InvalidInput(
            f"question exceeds {MAX_QUESTION_CHARS} characters"
        )
    return cleaned
