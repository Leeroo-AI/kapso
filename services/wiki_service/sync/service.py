"""
Main sync service entry point.

Coordinates bidirectional sync between local files and MediaWiki:
- File watcher for local -> wiki sync
- RecentChanges poller for wiki -> local sync
"""

import asyncio
import logging
import signal
import sys
import threading
from pathlib import Path
from typing import Optional

from .config import SyncConfig
from .state_manager import StateManager
from .sync_engine import SyncEngine
from .file_watcher import FileWatcher
from .wiki_poller import WikiPoller

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class WikiSyncService:
    """
    Main service coordinating bidirectional wiki sync.

    Runs both file watching (local->wiki) and RC polling (wiki->local)
    in a single asyncio event loop.
    """

    def __init__(self, config: Optional[SyncConfig] = None):
        self.config = config or SyncConfig()
        self.state = StateManager(self.config.sync_state)
        self.engine = SyncEngine(self.config, self.state)

        self._file_watcher: Optional[FileWatcher] = None
        self._poller: Optional[WikiPoller] = None
        self._running = False
        self._poll_task: Optional[asyncio.Task] = None

        # Track recently synced files to avoid echo
        self._recently_synced_local: set[str] = set()
        self._recently_synced_wiki: set[str] = set()

        # Debounced cache refresh - waits for a quiet period after syncs
        # before running a single Cargo + Main Page refresh
        self._refresh_timer: Optional[threading.Timer] = None
        self._refresh_lock = threading.Lock()

    def _on_local_change(self, file_path: Path) -> None:
        """Handle local file change."""
        rel_path = str(file_path.relative_to(self.config.wiki_dir))

        # Skip if this was just synced from wiki
        if rel_path in self._recently_synced_wiki:
            self._recently_synced_wiki.discard(rel_path)
            return

        logger.debug(f"Local file changed: {file_path}")
        if self.engine.sync_local_to_wiki(file_path):
            self._recently_synced_local.add(rel_path)
            self._schedule_refresh()

    def _on_local_delete(self, file_path: Path) -> None:
        """Handle local file deletion."""
        logger.debug(f"Local file deleted: {file_path}")
        self.engine.handle_local_delete(file_path)
        self._schedule_refresh()

    def _schedule_refresh(self) -> None:
        """
        Schedule a debounced Cargo + Main Page cache refresh.

        Resets the timer on each call, so the actual refresh only runs
        after `refresh_delay` seconds of quiet (no new syncs).
        This avoids running cargoRecreateData.php once per page during
        bulk imports — instead it runs once after all pages are done.
        """
        with self._refresh_lock:
            if self._refresh_timer is not None:
                self._refresh_timer.cancel()
            self._refresh_timer = threading.Timer(
                self.config.refresh_delay,
                self._do_refresh,
            )
            self._refresh_timer.daemon = True
            self._refresh_timer.start()

    def _do_refresh(self) -> None:
        """Run a single batched Cargo table rebuild + Main Page cache purge."""
        with self._refresh_lock:
            self._refresh_timer = None
        try:
            logger.info("Running batched wiki cache refresh...")
            self.engine._refresh_wiki_caches()
            logger.info("Wiki cache refresh complete")
        except Exception as e:
            logger.error(f"Wiki cache refresh failed: {e}")

    def _on_wiki_change(self, title: str, revid: int, content: str) -> None:
        """Handle wiki page change."""
        # Check if this was just synced from local
        file_path = self.engine.state.find_file_by_title(title)
        if file_path and file_path in self._recently_synced_local:
            self._recently_synced_local.discard(file_path)
            return

        logger.debug(f"Wiki page changed: {title} (rev {revid})")
        if self.engine.sync_wiki_to_local(title, revid, content):
            from .transforms import title_to_path
            local_path = title_to_path(self.config.wiki_dir, title)
            if local_path:
                rel_path = str(local_path.relative_to(self.config.wiki_dir))
                self._recently_synced_wiki.add(rel_path)

    def _on_wiki_delete(self, title: str) -> None:
        """Handle wiki page deletion."""
        logger.debug(f"Wiki page deleted: {title}")
        self.engine.handle_wiki_delete(title)

    async def _poll_loop(self) -> None:
        """Polling loop for wiki changes."""
        while self._running:
            try:
                self._poller.poll()
            except Exception as e:
                logger.error(f"Poll error: {e}")

            await asyncio.sleep(self.config.poll_interval)

    async def run(self) -> None:
        """Run the sync service."""
        logger.info("Starting Wiki Sync Service")
        logger.info(f"  Wiki URL: {self.config.wiki_url}")
        logger.info(f"  Wiki dir: {self.config.wiki_dir}")
        logger.info(f"  State file: {self.config.sync_state}")
        logger.info(f"  Poll interval: {self.config.poll_interval}s")
        logger.info(f"  Debounce: {self.config.debounce_ms}ms")
        logger.info(f"  Refresh delay: {self.config.refresh_delay}s")
        logger.info(f"  Conflict mode: {self.config.conflict_mode}")
        logger.info(f"  Delete mode: {self.config.delete_mode}")

        # Verify wiki directory exists
        if not self.config.wiki_dir.exists():
            logger.error(f"Wiki directory not found: {self.config.wiki_dir}")
            sys.exit(1)

        # Initialize components
        self._file_watcher = FileWatcher(
            wiki_dir=self.config.wiki_dir,
            debounce_ms=self.config.debounce_ms,
            on_change=self._on_local_change,
            on_delete=self._on_local_delete,
        )

        self._poller = WikiPoller(
            config=self.config,
            state_manager=self.state,
            on_page_change=self._on_wiki_change,
            on_page_delete=self._on_wiki_delete,
        )

        # Start file watcher
        self._file_watcher.start()

        # Start polling loop
        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop())

        logger.info("Sync service started. Press Ctrl+C to stop.")

        # Wait for shutdown signal
        try:
            await self._poll_task
        except asyncio.CancelledError:
            pass

    def stop(self) -> None:
        """Stop the sync service."""
        logger.info("Stopping sync service...")
        self._running = False

        if self._poll_task:
            self._poll_task.cancel()

        if self._file_watcher:
            self._file_watcher.stop()

        # Cancel any pending refresh timer
        with self._refresh_lock:
            if self._refresh_timer is not None:
                self._refresh_timer.cancel()
                self._refresh_timer = None

        logger.info("Sync service stopped")


def main():
    """Main entry point."""
    config = SyncConfig()
    service = WikiSyncService(config)

    # Handle shutdown signals
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def shutdown_handler():
        service.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown_handler)

    try:
        loop.run_until_complete(service.run())
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()


if __name__ == "__main__":
    main()
