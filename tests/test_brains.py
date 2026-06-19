"""Tests for brain filesystem CRUD (no LLM, no embeddings)."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from raggem import config
from raggem.core import brains as brains_mod
from raggem.security import InvalidInput


@pytest.fixture
def isolated_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect KNOWLEDGE_DIR / CHROMA_DB_DIR to a temp directory.

    Tests can write freely without polluting the developer's real data dir.
    """
    knowledge = tmp_path / "knowledge"
    chroma = tmp_path / "chroma"
    knowledge.mkdir()
    chroma.mkdir()
    monkeypatch.setattr(config, "KNOWLEDGE_DIR", knowledge)
    monkeypatch.setattr(config, "CHROMA_DB_DIR", chroma)
    monkeypatch.setattr(brains_mod, "KNOWLEDGE_DIR", knowledge)
    monkeypatch.setattr(brains_mod, "CHROMA_DB_DIR", chroma)
    return tmp_path


class TestBrainPaths:
    def test_resolves_inside_data_dir(self, isolated_data: Path) -> None:
        paths = brains_mod.brain_paths("alpha")
        assert paths.knowledge_dir.parent == (isolated_data / "knowledge").resolve()
        assert paths.chroma_dir.parent == (isolated_data / "chroma").resolve()

    def test_rejects_bad_brain_id(self, isolated_data: Path) -> None:
        with pytest.raises(InvalidInput):
            brains_mod.brain_paths("../escape")


class TestListBrains:
    def test_empty(self, isolated_data: Path) -> None:
        assert brains_mod.list_brains() == []

    def test_lists_existing(self, isolated_data: Path) -> None:
        (isolated_data / "knowledge" / "alpha").mkdir()
        (isolated_data / "knowledge" / "alpha" / "f.pdf").write_text("x")
        (isolated_data / "chroma" / "beta").mkdir()
        (isolated_data / "chroma" / "beta" / "marker").write_text("x")
        assert brains_mod.list_brains() == ["alpha", "beta"]

    def test_ignores_malformed_dirnames(self, isolated_data: Path) -> None:
        # A directory whose name isn't a valid brain id should not appear.
        (isolated_data / "knowledge" / "Bad Name").mkdir()
        (isolated_data / "knowledge" / "Bad Name" / "f.pdf").write_text("x")
        (isolated_data / "knowledge" / "ok").mkdir()
        (isolated_data / "knowledge" / "ok" / "f.pdf").write_text("x")
        assert brains_mod.list_brains() == ["ok"]


class TestListFiles:
    def test_filters_unsupported_extensions(self, isolated_data: Path) -> None:
        d = isolated_data / "knowledge" / "alpha"
        d.mkdir()
        (d / "doc.pdf").write_text("x")
        (d / "notes.md").write_text("x")
        (d / "readme.txt").write_text("x")
        (d / "script.exe").write_text("x")        # excluded
        (d / ".hidden.pdf").write_text("x")       # excluded
        names = [f.filename for f in brains_mod.list_files("alpha")]
        assert sorted(names) == ["doc.pdf", "notes.md", "readme.txt"]

    def test_empty_when_brain_missing(self, isolated_data: Path) -> None:
        assert brains_mod.list_files("alpha") == []


class TestUpload:
    def test_save_and_list_round_trip(self, isolated_data: Path) -> None:
        stream = io.BytesIO(b"hello world")
        out = brains_mod.save_uploaded_file("alpha", "doc.txt", stream)
        assert out.name == "doc.txt"
        assert out.read_text() == "hello world"
        assert [f.filename for f in brains_mod.list_files("alpha")] == ["doc.txt"]

    def test_rejects_oversize(
        self, isolated_data: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Shrink the cap so a small upload is "too big".
        monkeypatch.setattr(brains_mod, "MAX_UPLOAD_BYTES", 5)
        stream = io.BytesIO(b"this is more than five bytes")
        with pytest.raises(ValueError):
            brains_mod.save_uploaded_file("alpha", "doc.txt", stream)
        # No partial leftover.
        assert brains_mod.list_files("alpha") == []

    def test_rejects_path_traversal_in_filename(self, isolated_data: Path) -> None:
        stream = io.BytesIO(b"x")
        # The sanitiser strips path components, so the file ends up as
        # passwd.txt under the brain dir - never outside it.
        out = brains_mod.save_uploaded_file(
            "alpha", "../../etc/passwd.txt", stream
        )
        assert out.name == "passwd.txt"
        assert "knowledge/alpha" in str(out)


class TestDelete:
    def test_returns_false_if_missing(self, isolated_data: Path) -> None:
        (isolated_data / "knowledge" / "alpha").mkdir()
        assert brains_mod.delete_file("alpha", "nope.pdf") is False

    def test_deletes_existing(self, isolated_data: Path) -> None:
        d = isolated_data / "knowledge" / "alpha"
        d.mkdir()
        (d / "doc.pdf").write_text("x")
        assert brains_mod.delete_file("alpha", "doc.pdf") is True
        assert not (d / "doc.pdf").exists()


class TestDestroy:
    def test_removes_both_dirs(self, isolated_data: Path) -> None:
        kd = isolated_data / "knowledge" / "alpha"
        cd = isolated_data / "chroma" / "alpha"
        kd.mkdir()
        cd.mkdir()
        (kd / "doc.pdf").write_text("x")
        (cd / "v").write_text("x")
        brains_mod.destroy_brain("alpha")
        assert not kd.exists()
        assert not cd.exists()

    def test_idempotent_on_missing(self, isolated_data: Path) -> None:
        # No exception when destroying something that doesn't exist.
        brains_mod.destroy_brain("nonexistent")
