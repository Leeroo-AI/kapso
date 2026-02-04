"""Tests for state manager."""

import json
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from ..state_manager import (
    StateManager,
    SyncState,
    FileState,
    ConflictInfo,
    compute_content_hash,
)


class TestComputeContentHash:
    """Test content hash computation."""

    def test_consistent_hash(self):
        content = "test content"
        hash1 = compute_content_hash(content)
        hash2 = compute_content_hash(content)
        assert hash1 == hash2

    def test_hash_format(self):
        hash_val = compute_content_hash("test")
        assert hash_val.startswith("sha256:")
        assert len(hash_val) == 7 + 64  # "sha256:" + 64 hex chars

    def test_different_content_different_hash(self):
        hash1 = compute_content_hash("content1")
        hash2 = compute_content_hash("content2")
        assert hash1 != hash2


class TestStateManager:
    """Test StateManager functionality."""

    def test_load_creates_empty_state(self):
        with TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            manager = StateManager(state_path)
            state = manager.load()

            assert state.version == 1
            assert state.last_rc_timestamp is None
            assert state.last_rc_id is None
            assert state.files == {}
            assert state.pending == []
            assert state.conflicts == []

    def test_save_and_load(self):
        with TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            manager = StateManager(state_path)

            # Load initial state
            manager.load()

            # Update file state
            manager.update_file_state(
                rel_path="heuristics/Test.md",
                content_hash="sha256:abc123",
                wiki_title="Heuristic:Test",
                wiki_revid=42,
            )

            # Create new manager and load
            manager2 = StateManager(state_path)
            state = manager2.load()

            assert "heuristics/Test.md" in state.files
            file_state = state.files["heuristics/Test.md"]
            assert file_state.content_hash == "sha256:abc123"
            assert file_state.wiki_title == "Heuristic:Test"
            assert file_state.wiki_revid == 42

    def test_update_rc_position(self):
        with TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            manager = StateManager(state_path)
            manager.load()

            manager.update_rc_position(
                timestamp="2024-01-01T00:00:00Z",
                rc_id=12345,
            )

            state = manager.load()
            assert state.last_rc_timestamp == "2024-01-01T00:00:00Z"
            assert state.last_rc_id == 12345

    def test_add_and_remove_conflict(self):
        with TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            manager = StateManager(state_path)
            manager.load()

            conflict = ConflictInfo(
                file_path="heuristics/Test.md",
                wiki_title="Heuristic:Test",
                local_hash="sha256:local",
                wiki_revid=42,
                detected_at="2024-01-01T00:00:00Z",
            )
            manager.add_conflict(conflict)

            conflicts = manager.get_conflicts()
            assert len(conflicts) == 1
            assert conflicts[0].file_path == "heuristics/Test.md"

            manager.remove_conflict("heuristics/Test.md")
            conflicts = manager.get_conflicts()
            assert len(conflicts) == 0

    def test_find_file_by_title(self):
        with TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            manager = StateManager(state_path)
            manager.load()

            manager.update_file_state(
                rel_path="heuristics/Test.md",
                content_hash="sha256:abc",
                wiki_title="Heuristic:Test",
                wiki_revid=1,
            )

            found = manager.find_file_by_title("Heuristic:Test")
            assert found == "heuristics/Test.md"

            not_found = manager.find_file_by_title("Heuristic:NotExists")
            assert not_found is None

    def test_remove_file_state(self):
        with TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            manager = StateManager(state_path)
            manager.load()

            manager.update_file_state(
                rel_path="heuristics/Test.md",
                content_hash="sha256:abc",
                wiki_title="Heuristic:Test",
                wiki_revid=1,
            )

            assert manager.get_file_state("heuristics/Test.md") is not None

            manager.remove_file_state("heuristics/Test.md")
            assert manager.get_file_state("heuristics/Test.md") is None
