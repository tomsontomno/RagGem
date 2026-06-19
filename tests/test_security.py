"""Tests for input validation in :mod:`raggem.security`."""

from __future__ import annotations

from pathlib import Path

import pytest

from raggem.security import (
    InvalidInput,
    safe_join,
    sanitize_filename,
    sanitize_question,
    validate_brain_id,
)


# ---------------------------------------------------------------------------
# Brain IDs
# ---------------------------------------------------------------------------
class TestValidateBrainId:
    @pytest.mark.parametrize(
        "value",
        ["a", "default", "site-1", "my_brain", "abc123", "x" * 64],
    )
    def test_accepts_safe_ids(self, value: str) -> None:
        assert validate_brain_id(value) == value

    @pytest.mark.parametrize(
        "value",
        [
            "",                # empty
            "_leading",        # cannot start with underscore
            "-leading",        # cannot start with dash
            "UPPER",           # no uppercase
            "with space",      # no whitespace
            "with/slash",      # no path separators
            "with.dot",        # no dots
            "with:colon",      # no colons
            "..",              # path traversal classic
            "../etc",          # path traversal
            "x" * 65,          # too long
            "a\nb",            # newline
            "a\x00b",           # null byte
        ],
    )
    def test_rejects_unsafe_ids(self, value: str) -> None:
        with pytest.raises(InvalidInput):
            validate_brain_id(value)


# ---------------------------------------------------------------------------
# Filenames
# ---------------------------------------------------------------------------
class TestSanitizeFilename:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("doc.pdf", "doc.pdf"),
            ("My Document.PDF", "My Document.PDF"),
            ("notes.md", "notes.md"),
            ("readme.txt", "readme.txt"),
            ("foo.markdown", "foo.markdown"),
            # Path components are stripped:
            ("../../etc/passwd.txt", "passwd.txt"),
            ("/absolute/path/doc.pdf", "doc.pdf"),
            ("  spaced.pdf  ", "spaced.pdf"),
        ],
    )
    def test_strips_path_components(self, raw: str, expected: str) -> None:
        assert sanitize_filename(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        [
            "",
            ".",
            "..",
            ".hidden.pdf",       # leading dot
            "doc.exe",            # unsupported extension
            "doc",                # no extension
            "doc.PDF.bak",        # unsupported extension (.bak)
            "weird\x00.pdf",     # null byte
            "a" * 250 + ".pdf",  # too long
            "doc<.pdf",           # forbidden character
            "doc>.pdf",
            "doc:.pdf",
            "doc|.pdf",
            "doc?.pdf",
            "doc*.pdf",
            "doc\"quote.pdf",
        ],
    )
    def test_rejects_unsafe(self, raw: str) -> None:
        with pytest.raises(InvalidInput):
            sanitize_filename(raw)

    def test_collapses_whitespace(self) -> None:
        assert sanitize_filename("a  b   c.pdf") == "a b c.pdf"

    def test_handles_unicode(self) -> None:
        # NFKC normalises the full-width digit to ASCII; remains valid.
        assert sanitize_filename("doc１.pdf") == "doc1.pdf"


# ---------------------------------------------------------------------------
# safe_join
# ---------------------------------------------------------------------------
class TestSafeJoin:
    def test_joins_inside_root(self, tmp_path: Path) -> None:
        out = safe_join(tmp_path, "sub", "file.pdf")
        assert tmp_path in out.parents

    def test_rejects_parent_traversal(self, tmp_path: Path) -> None:
        with pytest.raises(InvalidInput):
            safe_join(tmp_path, "..", "etc")

    def test_rejects_absolute_override(self, tmp_path: Path) -> None:
        # Passing an absolute path as a join component would on raw
        # ``os.path.join`` override the root. safe_join must reject it.
        with pytest.raises(InvalidInput):
            safe_join(tmp_path, "/etc/passwd")

    def test_root_itself_is_ok(self, tmp_path: Path) -> None:
        # Joining nothing is fine.
        assert safe_join(tmp_path) == tmp_path.resolve()


# ---------------------------------------------------------------------------
# Question sanitisation
# ---------------------------------------------------------------------------
class TestSanitizeQuestion:
    def test_passes_normal_question(self) -> None:
        assert sanitize_question("What is the main topic?") == "What is the main topic?"

    def test_strips_whitespace(self) -> None:
        assert sanitize_question("   hello\n  ") == "hello"

    def test_rejects_empty(self) -> None:
        with pytest.raises(InvalidInput):
            sanitize_question("")
        with pytest.raises(InvalidInput):
            sanitize_question("   ")

    def test_rejects_too_long(self) -> None:
        with pytest.raises(InvalidInput):
            sanitize_question("x" * 10_000)

    def test_strips_control_characters(self) -> None:
        # Carriage returns and other control chars get filtered, leaving
        # the visible content intact.
        assert sanitize_question("hi\x07\x08there") == "hithere"

    def test_keeps_newlines_and_tabs(self) -> None:
        # Multi-line questions are fine for embedding.
        assert "\n" in sanitize_question("line1\nline2")
