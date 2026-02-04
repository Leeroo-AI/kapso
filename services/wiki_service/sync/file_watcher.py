"""
File watcher with debouncing for local wiki files.

Uses watchdog to monitor file changes and debounces rapid saves
(e.g., from editors that save multiple times quickly).
"""

import logging
import threading
import time
from pathlib import Path
from typing import Callable, Optional
from watchdog.observers import Observer
from watchdog.events import (
    FileSystemEventHandler,
    FileCreatedEvent,
    FileModifiedEvent,
    FileDeletedEvent,
    FileMovedEvent,
)

from .transforms import is_synced_path

logger = logging.getLogger(__name__)


class DebouncedHandler(FileSystemEventHandler):
    """
    File event handler with debouncing.

    Collects file changes and only triggers callbacks after a debounce
    period of no changes (to handle editors that save multiple times).
    """

    def __init__(
        self,
        wiki_dir: Path,
        debounce_ms: int,
        on_change: Callable[[Path], None],
        on_delete: Callable[[Path], None],
    ):
        self.wiki_dir = wiki_dir
        self.debounce_seconds = debounce_ms / 1000.0
        self.on_change = on_change
        self.on_delete = on_delete

        self._pending: dict[str, tuple[str, float]] = {}  # path -> (event_type, timestamp)
        self._lock = threading.Lock()
        self._timer: Optional[threading.Timer] = None

    def _schedule_flush(self) -> None:
        """Schedule a flush of pending events after debounce period."""
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self.debounce_seconds, self._flush)
            self._timer.daemon = True
            self._timer.start()

    def _flush(self) -> None:
        """Process all pending events."""
        with self._lock:
            pending = dict(self._pending)
            self._pending.clear()
            self._timer = None

        for path_str, (event_type, _timestamp) in pending.items():
            path = Path(path_str)
            try:
                if event_type == "delete":
                    self.on_delete(path)
                else:
                    # For create/modify, check if file still exists
                    if path.exists():
                        self.on_change(path)
            except Exception as e:
                logger.error(f"Error handling {event_type} for {path}: {e}")

    def _add_pending(self, path: Path, event_type: str) -> None:
        """Add a pending event, updating existing if present."""
        if not path.suffix == ".md":
            return
        if not is_synced_path(self.wiki_dir, path):
            return

        with self._lock:
            self._pending[str(path)] = (event_type, time.time())

        self._schedule_flush()

    def on_created(self, event: FileCreatedEvent) -> None:
        if event.is_directory:
            return
        self._add_pending(Path(event.src_path), "create")

    def on_modified(self, event: FileModifiedEvent) -> None:
        if event.is_directory:
            return
        self._add_pending(Path(event.src_path), "modify")

    def on_deleted(self, event: FileDeletedEvent) -> None:
        if event.is_directory:
            return
        # Only track .md files
        if not event.src_path.endswith(".md"):
            return
        self._add_pending(Path(event.src_path), "delete")

    def on_moved(self, event: FileMovedEvent) -> None:
        if event.is_directory:
            return
        # Treat move as delete of old + create of new
        if event.src_path.endswith(".md"):
            self._add_pending(Path(event.src_path), "delete")
        if event.dest_path.endswith(".md"):
            self._add_pending(Path(event.dest_path), "create")


class FileWatcher:
    """
    Watches wiki directory for file changes.

    Provides debounced callbacks for file modifications and deletions.
    """

    def __init__(
        self,
        wiki_dir: Path,
        debounce_ms: int = 500,
        on_change: Optional[Callable[[Path], None]] = None,
        on_delete: Optional[Callable[[Path], None]] = None,
    ):
        self.wiki_dir = wiki_dir
        self.debounce_ms = debounce_ms
        self.on_change = on_change or (lambda p: None)
        self.on_delete = on_delete or (lambda p: None)

        self._observer: Optional[Observer] = None
        self._handler: Optional[DebouncedHandler] = None

    def start(self) -> None:
        """Start watching the wiki directory."""
        if self._observer is not None:
            return

        self._handler = DebouncedHandler(
            wiki_dir=self.wiki_dir,
            debounce_ms=self.debounce_ms,
            on_change=self.on_change,
            on_delete=self.on_delete,
        )

        self._observer = Observer()
        self._observer.schedule(
            self._handler,
            str(self.wiki_dir),
            recursive=True,
        )
        self._observer.start()
        logger.info(f"File watcher started for {self.wiki_dir}")

    def stop(self) -> None:
        """Stop watching the wiki directory."""
        if self._observer is None:
            return

        self._observer.stop()
        self._observer.join(timeout=5)
        self._observer = None
        self._handler = None
        logger.info("File watcher stopped")

    def __enter__(self) -> "FileWatcher":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
