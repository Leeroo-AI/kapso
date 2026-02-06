"""
Wiki RecentChanges poller for wiki sync.

Polls MediaWiki's RecentChanges API to detect page edits
and trigger synchronization to local files.
"""

import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Callable, Optional

# Add parent dir to path for mw_client import
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
from mw_client import MWClient

from .config import SyncConfig
from .state_manager import StateManager
from .transforms import is_synced_title, get_synced_namespaces, NAMESPACE_TO_FOLDER

logger = logging.getLogger(__name__)


# Namespace IDs for our synced namespaces
# These are defined in LocalSettings.php and registered during wiki setup
# Principle=3000, Workflow=3002, Implementation=3004, Heuristic=3008, Environment=3010
NAMESPACE_IDS = {
    "Principle": 3000,
    "Workflow": 3002,
    "Implementation": 3004,
    "Heuristic": 3008,
    "Environment": 3010,
}


def backoff_timestamp(ts_iso: str, seconds: int = 1) -> str:
    """Back off timestamp by given seconds to handle clock skew."""
    dt = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
    return (dt - timedelta(seconds=seconds)).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class WikiPoller:
    """
    Polls MediaWiki RecentChanges for page edits.

    Detects new/edited/deleted pages in synced namespaces and
    triggers callbacks to sync them to local files.
    """

    def __init__(
        self,
        config: SyncConfig,
        state_manager: StateManager,
        on_page_change: Callable[[str, int, str], None],  # title, revid, content
        on_page_delete: Callable[[str], None],  # title
    ):
        self.config = config
        self.state = state_manager
        self.on_page_change = on_page_change
        self.on_page_delete = on_page_delete
        self._mw: Optional[MWClient] = None

    @property
    def mw(self) -> MWClient:
        """Lazy-initialize MediaWiki client."""
        if self._mw is None:
            self._mw = MWClient(
                self.config.wiki_url,
                self.config.mw_user,
                self.config.mw_pass,
            )
            self._mw.login()
            logger.info(f"Wiki poller connected to {self._mw.api_url}")
        return self._mw

    def _get_namespace_filter(self) -> str:
        """Get namespace IDs as pipe-separated string for API filter."""
        ids = [str(NAMESPACE_IDS[ns]) for ns in get_synced_namespaces() if ns in NAMESPACE_IDS]
        return "|".join(ids)

    def poll(self) -> int:
        """
        Poll for recent changes since last sync.

        Returns number of changes processed.
        """
        state = self.state.load()
        last_ts = state.last_rc_timestamp
        last_rcid = state.last_rc_id

        # Build query parameters
        params = {
            "action": "query",
            "list": "recentchanges",
            "rcprop": "title|ids|timestamp|loginfo",
            "rclimit": "500",
            "rctype": "edit|new|log",
            "format": "json",
        }

        # Filter to synced namespaces
        ns_filter = self._get_namespace_filter()
        if ns_filter:
            params["rcnamespace"] = ns_filter

        # Start from last known position
        if last_ts:
            params["rcstart"] = backoff_timestamp(last_ts, 1)
            params["rcdir"] = "newer"

        changes_processed = 0
        latest_ts = last_ts
        latest_rcid = last_rcid

        # Page changes to process (deduplicate by pageid)
        page_changes: dict[int, dict] = {}  # pageid -> {title, revid, timestamp}
        page_deletes: set[str] = set()  # titles to delete

        try:
            while True:
                result = self.mw._api(params=params)
                changes = result.get("query", {}).get("recentchanges", [])

                if not changes:
                    break

                for change in changes:
                    rcid = change.get("rcid")
                    title = change.get("title", "")
                    pageid = change.get("pageid", 0)
                    revid = change.get("revid", 0)
                    timestamp = change.get("timestamp", "")
                    rc_type = change.get("type", "")
                    log_type = change.get("logtype", "")

                    # Skip if we've already seen this rcid
                    if last_rcid is not None and rcid <= last_rcid:
                        continue

                    # Update watermarks
                    if rcid and (latest_rcid is None or rcid > latest_rcid):
                        latest_rcid = rcid
                    if timestamp and (latest_ts is None or timestamp > latest_ts):
                        latest_ts = timestamp

                    # Skip if not in synced namespace
                    if not is_synced_title(title):
                        continue

                    # Handle delete
                    if log_type == "delete" or rc_type == "log" and change.get("logaction") == "delete":
                        page_deletes.add(title)
                        page_changes.pop(pageid, None)
                        continue

                    # Handle edit/new
                    if rc_type in ("edit", "new") and pageid > 0:
                        current = page_changes.get(pageid, {})
                        if not current or (revid and revid > current.get("revid", 0)):
                            page_changes[pageid] = {
                                "title": title,
                                "revid": revid,
                                "timestamp": timestamp,
                            }

                # Check for continuation
                cont = result.get("continue", {}).get("rccontinue")
                if cont:
                    params["rccontinue"] = cont
                else:
                    break

        except Exception as e:
            logger.error(f"Error polling recent changes: {e}")
            return 0

        # Process deletes first
        for title in page_deletes:
            try:
                self.on_page_delete(title)
                changes_processed += 1
            except Exception as e:
                logger.error(f"Error handling delete for {title}: {e}")

        # Process page changes
        for pageid, meta in page_changes.items():
            title = meta["title"]
            revid = meta["revid"]

            # Skip if deleted in same batch
            if title in page_deletes:
                continue

            try:
                # Fetch current content
                content = self._get_page_content(title)
                if content is not None:
                    self.on_page_change(title, revid, content)
                    changes_processed += 1
            except Exception as e:
                logger.error(f"Error syncing {title}: {e}")

        # Update state with new position
        if latest_ts or latest_rcid:
            self.state.update_rc_position(
                timestamp=latest_ts,
                rc_id=latest_rcid,
            )

        if changes_processed > 0:
            logger.info(f"Processed {changes_processed} wiki changes")

        return changes_processed

    def _get_page_content(self, title: str) -> Optional[str]:
        """Get current wikitext content of a page."""
        try:
            params = {
                "action": "query",
                "titles": title,
                "prop": "revisions",
                "rvprop": "content",
                "rvslots": "main",
                "format": "json",
            }
            result = self.mw._api(params=params)
            pages = result.get("query", {}).get("pages", {})
            for page_id, page_data in pages.items():
                if page_id == "-1":
                    return None
                revisions = page_data.get("revisions", [])
                if revisions:
                    return revisions[0].get("slots", {}).get("main", {}).get("*", "")
            return None
        except Exception as e:
            logger.error(f"Failed to get content for {title}: {e}")
            return None
