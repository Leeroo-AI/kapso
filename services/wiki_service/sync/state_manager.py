"""
State manager for wiki sync service.

Handles persistence of sync state including content hashes, wiki revision IDs,
and tracking of pending changes and conflicts.
"""

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import threading


@dataclass
class FileState:
    """State for a single synced file."""
    content_hash: str  # sha256 hash of file content
    wiki_title: str
    wiki_revid: int
    synced_at: str  # ISO timestamp


@dataclass
class ConflictInfo:
    """Information about a detected conflict."""
    file_path: str
    wiki_title: str
    local_hash: str
    wiki_revid: int
    detected_at: str
    local_conflict_file: Optional[str] = None
    wiki_conflict_file: Optional[str] = None


@dataclass
class SyncState:
    """Full sync state."""
    version: int = 1
    last_rc_timestamp: Optional[str] = None
    last_rc_id: Optional[int] = None
    files: dict[str, FileState] = field(default_factory=dict)
    pending: list[str] = field(default_factory=list)
    conflicts: list[ConflictInfo] = field(default_factory=list)


class StateManager:
    """Manages persistent sync state."""

    def __init__(self, state_path: Path):
        self.state_path = state_path
        self._lock = threading.RLock()  # Use reentrant lock to allow nested calls
        self._state: Optional[SyncState] = None

    def load(self) -> SyncState:
        """Load state from disk, or create empty state if not exists."""
        with self._lock:
            if self._state is not None:
                return self._state

            if self.state_path.exists():
                try:
                    data = json.loads(self.state_path.read_text())
                    self._state = self._parse_state(data)
                except Exception:
                    self._state = SyncState()
            else:
                self._state = SyncState()

            return self._state

    def _parse_state(self, data: dict) -> SyncState:
        """Parse state from JSON dict."""
        files = {}
        for path, file_data in data.get("files", {}).items():
            files[path] = FileState(
                content_hash=file_data.get("content_hash", ""),
                wiki_title=file_data.get("wiki_title", ""),
                wiki_revid=file_data.get("wiki_revid", 0),
                synced_at=file_data.get("synced_at", ""),
            )

        conflicts = []
        for conflict_data in data.get("conflicts", []):
            conflicts.append(ConflictInfo(
                file_path=conflict_data.get("file_path", ""),
                wiki_title=conflict_data.get("wiki_title", ""),
                local_hash=conflict_data.get("local_hash", ""),
                wiki_revid=conflict_data.get("wiki_revid", 0),
                detected_at=conflict_data.get("detected_at", ""),
                local_conflict_file=conflict_data.get("local_conflict_file"),
                wiki_conflict_file=conflict_data.get("wiki_conflict_file"),
            ))

        return SyncState(
            version=data.get("version", 1),
            last_rc_timestamp=data.get("last_rc_timestamp"),
            last_rc_id=data.get("last_rc_id"),
            files=files,
            pending=data.get("pending", []),
            conflicts=conflicts,
        )

    def save(self) -> None:
        """Save current state to disk."""
        with self._lock:
            if self._state is None:
                return

            self.state_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert to dict for JSON serialization
            data = {
                "version": self._state.version,
                "last_rc_timestamp": self._state.last_rc_timestamp,
                "last_rc_id": self._state.last_rc_id,
                "files": {
                    path: asdict(file_state)
                    for path, file_state in self._state.files.items()
                },
                "pending": self._state.pending,
                "conflicts": [asdict(c) for c in self._state.conflicts],
            }

            self.state_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    def get_file_state(self, rel_path: str) -> Optional[FileState]:
        """Get state for a specific file."""
        state = self.load()
        return state.files.get(rel_path)

    def update_file_state(
        self,
        rel_path: str,
        content_hash: str,
        wiki_title: str,
        wiki_revid: int,
    ) -> None:
        """Update state for a synced file."""
        with self._lock:
            state = self.load()
            state.files[rel_path] = FileState(
                content_hash=content_hash,
                wiki_title=wiki_title,
                wiki_revid=wiki_revid,
                synced_at=datetime.now(timezone.utc).isoformat(),
            )
            self.save()

    def remove_file_state(self, rel_path: str) -> None:
        """Remove state for a file (e.g., when deleted)."""
        with self._lock:
            state = self.load()
            state.files.pop(rel_path, None)
            self.save()

    def update_rc_position(
        self,
        timestamp: Optional[str] = None,
        rc_id: Optional[int] = None,
    ) -> None:
        """Update RecentChanges polling position."""
        with self._lock:
            state = self.load()
            if timestamp:
                state.last_rc_timestamp = timestamp
            if rc_id is not None:
                state.last_rc_id = rc_id
            self.save()

    def add_conflict(self, conflict: ConflictInfo) -> None:
        """Add a conflict to the state."""
        with self._lock:
            state = self.load()
            state.conflicts.append(conflict)
            self.save()

    def remove_conflict(self, file_path: str) -> None:
        """Remove a conflict by file path."""
        with self._lock:
            state = self.load()
            state.conflicts = [
                c for c in state.conflicts if c.file_path != file_path
            ]
            self.save()

    def get_conflicts(self) -> list[ConflictInfo]:
        """Get all current conflicts."""
        return self.load().conflicts

    def find_file_by_title(self, wiki_title: str) -> Optional[str]:
        """Find a file path by wiki title."""
        state = self.load()
        for rel_path, file_state in state.files.items():
            if file_state.wiki_title == wiki_title:
                return rel_path
        return None


def compute_content_hash(content: str) -> str:
    """Compute SHA256 hash of content."""
    return f"sha256:{hashlib.sha256(content.encode('utf-8')).hexdigest()}"
