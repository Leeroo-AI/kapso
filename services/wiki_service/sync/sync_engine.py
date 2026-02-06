"""
Core sync logic for bidirectional wiki sync.

Handles conflict detection, content synchronization, and conflict resolution.
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Add parent dir to path for mw_client import
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
from mw_client import MWClient

from .config import SyncConfig
from .state_manager import (
    StateManager,
    FileState,
    ConflictInfo,
    compute_content_hash,
)
from .transforms import (
    path_to_title,
    title_to_path,
    normalize_text,
    is_synced_title,
    is_synced_path,
    prepare_content_for_wiki,
    FOLDER_TO_NAMESPACE,
)

logger = logging.getLogger(__name__)


class SyncEngine:
    """Core sync logic for bidirectional wiki synchronization."""

    def __init__(self, config: SyncConfig, state_manager: StateManager):
        self.config = config
        self.state = state_manager
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
            logger.info(f"Connected to MediaWiki at {self._mw.api_url}")
        return self._mw

    def get_wiki_page_info(self, title: str) -> Optional[dict]:
        """
        Get page info including current revision ID.

        Returns dict with 'revid' and 'content' keys, or None if page doesn't exist.
        """
        try:
            params = {
                "action": "query",
                "titles": title,
                "prop": "revisions",
                "rvprop": "ids|content",
                "rvslots": "main",
                "format": "json",
            }
            result = self.mw._api(params=params)
            pages = result.get("query", {}).get("pages", {})
            for page_id, page_data in pages.items():
                if page_id == "-1":
                    return None  # Page doesn't exist
                revisions = page_data.get("revisions", [])
                if revisions:
                    rev = revisions[0]
                    content = rev.get("slots", {}).get("main", {}).get("*", "")
                    return {
                        "revid": rev.get("revid", 0),
                        "content": content,
                    }
            return None
        except Exception as e:
            logger.error(f"Failed to get page info for {title}: {e}")
            return None

    def sync_local_to_wiki(self, file_path: Path) -> bool:
        """
        Sync a local file change to the wiki.

        Returns True if sync succeeded, False if conflict detected or error.
        """
        if not is_synced_path(self.config.wiki_dir, file_path):
            return False

        title = path_to_title(self.config.wiki_dir, file_path)
        if not title:
            return False

        rel_path = str(file_path.relative_to(self.config.wiki_dir))

        # Read local content
        try:
            content = normalize_text(file_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
            return False

        if not content.strip():
            logger.warning(f"Skipping empty file: {file_path}")
            return False

        content_hash = compute_content_hash(content)

        # Derive namespace and page name for metadata injection
        # e.g. file in workflows/ folder -> namespace="Workflow", page_name="Foo_Bar"
        try:
            rel = file_path.relative_to(self.config.wiki_dir)
            folder_name = rel.parts[0] if rel.parts else ""
        except ValueError:
            folder_name = ""
        namespace = FOLDER_TO_NAMESPACE.get(folder_name)
        page_name = file_path.stem  # filename without .md

        # Transform content for MediaWiki rendering:
        # - Strip redundant H1 heading (would render as numbered list item)
        # - Convert [[source::Type|Label|URL]] to [URL Label] external links
        # - Add PageInfo Cargo template and Category tag
        wiki_content = prepare_content_for_wiki(content, namespace, page_name)

        # Check current state
        file_state = self.state.get_file_state(rel_path)

        # Get current wiki state
        wiki_info = self.get_wiki_page_info(title)

        # Detect conflict
        if file_state and wiki_info:
            # Both exist - check if wiki changed since last sync
            if wiki_info["revid"] > file_state.wiki_revid:
                # Wiki has newer revision
                if content_hash != file_state.content_hash:
                    # Local also changed - CONFLICT
                    logger.warning(f"Conflict detected for {rel_path}")
                    self._handle_conflict(
                        rel_path=rel_path,
                        title=title,
                        local_content=content,
                        wiki_content=wiki_info["content"],
                        wiki_revid=wiki_info["revid"],
                    )
                    return False
                else:
                    # Local unchanged, wiki changed - pull wiki version
                    logger.info(f"Wiki has newer version, skipping local push: {title}")
                    return False

        # Push to wiki (use transformed content, not raw local content)
        try:
            result = self.mw.edit(
                title,
                text=wiki_content,
                summary="Sync from local file",
                bot=True,
            )
            if result.get("edit", {}).get("result") == "Success":
                new_revid = result.get("edit", {}).get("newrevid", 0)
                self.state.update_file_state(
                    rel_path=rel_path,
                    content_hash=content_hash,
                    wiki_title=title,
                    wiki_revid=new_revid,
                )
                logger.info(f"Synced local->wiki: {title} (rev {new_revid})")
                return True
            else:
                logger.error(f"Wiki edit failed for {title}: {result}")
                return False
        except Exception as e:
            logger.error(f"Failed to sync {title} to wiki: {e}")
            return False

    def sync_wiki_to_local(self, title: str, revid: int, content: str) -> bool:
        """
        Sync a wiki change to local file.

        Returns True if sync succeeded, False if conflict detected or error.
        """
        if not is_synced_title(title):
            return False

        file_path = title_to_path(self.config.wiki_dir, title)
        if not file_path:
            return False

        rel_path = str(file_path.relative_to(self.config.wiki_dir))
        content = normalize_text(content)
        content_hash = compute_content_hash(content)

        # Check current state
        file_state = self.state.get_file_state(rel_path)

        # Check if local file exists and has changes
        if file_path.exists():
            try:
                local_content = normalize_text(
                    file_path.read_text(encoding="utf-8")
                )
                local_hash = compute_content_hash(local_content)
            except Exception as e:
                logger.error(f"Failed to read local file {file_path}: {e}")
                return False

            # Detect conflict
            if file_state and local_hash != file_state.content_hash:
                # Local has unsaved changes
                if revid > file_state.wiki_revid:
                    # Wiki also changed - CONFLICT
                    logger.warning(f"Conflict detected for {title}")
                    self._handle_conflict(
                        rel_path=rel_path,
                        title=title,
                        local_content=local_content,
                        wiki_content=content,
                        wiki_revid=revid,
                    )
                    return False

        # Write to local file
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
            self.state.update_file_state(
                rel_path=rel_path,
                content_hash=content_hash,
                wiki_title=title,
                wiki_revid=revid,
            )
            logger.info(f"Synced wiki->local: {title} -> {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to write local file {file_path}: {e}")
            return False

    def _handle_conflict(
        self,
        rel_path: str,
        title: str,
        local_content: str,
        wiki_content: str,
        wiki_revid: int,
    ) -> None:
        """Handle a sync conflict based on configured mode."""
        if self.config.conflict_mode == "wiki_wins":
            # Just overwrite local with wiki version
            file_path = self.config.wiki_dir / rel_path
            file_path.write_text(wiki_content, encoding="utf-8")
            self.state.update_file_state(
                rel_path=rel_path,
                content_hash=compute_content_hash(wiki_content),
                wiki_title=title,
                wiki_revid=wiki_revid,
            )
            logger.info(f"Conflict resolved (wiki_wins): {title}")
            return

        if self.config.conflict_mode == "local_wins":
            # Push local to wiki
            result = self.mw.edit(
                title,
                text=local_content,
                summary="Sync from local (conflict resolution: local_wins)",
                bot=True,
            )
            if result.get("edit", {}).get("result") == "Success":
                new_revid = result.get("edit", {}).get("newrevid", 0)
                self.state.update_file_state(
                    rel_path=rel_path,
                    content_hash=compute_content_hash(local_content),
                    wiki_title=title,
                    wiki_revid=new_revid,
                )
                logger.info(f"Conflict resolved (local_wins): {title}")
            return

        # Default: conflict_mode == "file" - create conflict files
        self._create_conflict_files(
            rel_path=rel_path,
            title=title,
            local_content=local_content,
            wiki_content=wiki_content,
            wiki_revid=wiki_revid,
        )

    def _create_conflict_files(
        self,
        rel_path: str,
        title: str,
        local_content: str,
        wiki_content: str,
        wiki_revid: int,
    ) -> None:
        """Create conflict files in _conflicts/ directory."""
        conflicts_dir = self.config.conflicts_dir
        conflicts_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        base_name = Path(rel_path).stem

        local_file = conflicts_dir / f"{base_name}.LOCAL.{timestamp}.md"
        wiki_file = conflicts_dir / f"{base_name}.WIKI.{timestamp}.md"
        info_file = conflicts_dir / f"{base_name}.CONFLICT.{timestamp}.json"

        local_file.write_text(local_content, encoding="utf-8")
        wiki_file.write_text(wiki_content, encoding="utf-8")

        import json
        info_file.write_text(json.dumps({
            "file_path": rel_path,
            "wiki_title": title,
            "wiki_revid": wiki_revid,
            "detected_at": datetime.now(timezone.utc).isoformat(),
            "local_file": str(local_file),
            "wiki_file": str(wiki_file),
        }, indent=2))

        # Record conflict in state
        self.state.add_conflict(ConflictInfo(
            file_path=rel_path,
            wiki_title=title,
            local_hash=compute_content_hash(local_content),
            wiki_revid=wiki_revid,
            detected_at=datetime.now(timezone.utc).isoformat(),
            local_conflict_file=str(local_file),
            wiki_conflict_file=str(wiki_file),
        ))

        logger.warning(
            f"Conflict files created for {title}:\n"
            f"  Local: {local_file}\n"
            f"  Wiki:  {wiki_file}\n"
            f"  Info:  {info_file}"
        )

    def handle_local_delete(self, file_path: Path) -> None:
        """Handle deletion of a local file."""
        if not is_synced_path(self.config.wiki_dir, file_path):
            return

        rel_path = str(file_path.relative_to(self.config.wiki_dir))
        file_state = self.state.get_file_state(rel_path)

        # Get title from state or derive from path
        if file_state:
            title = file_state.wiki_title
        else:
            # File not tracked - derive title from path anyway
            title = path_to_title(self.config.wiki_dir, file_path)
            if not title:
                return
            logger.debug(f"Untracked file deleted, derived title: {title}")

        if self.config.delete_mode == "log":
            logger.warning(
                f"Local file deleted: {rel_path} (wiki: {title}). "
                f"Manual action required to delete wiki page."
            )
            if file_state:
                self.state.remove_file_state(rel_path)
        elif self.config.delete_mode == "sync":
            try:
                self.mw.delete(title, reason="Sync: local file deleted")
                logger.info(f"Deleted wiki page: {title}")
            except Exception as e:
                logger.error(f"Failed to delete wiki page {title}: {e}")
            if file_state:
                self.state.remove_file_state(rel_path)
        elif self.config.delete_mode == "archive":
            # Move to archive namespace instead of deleting
            logger.warning(
                f"Archive mode not implemented. "
                f"Local file deleted: {rel_path} (wiki: {title})"
            )
            if file_state:
                self.state.remove_file_state(rel_path)

    def handle_wiki_delete(self, title: str) -> None:
        """Handle deletion of a wiki page."""
        if not is_synced_title(title):
            return

        rel_path = self.state.find_file_by_title(title)
        if not rel_path:
            return

        file_path = self.config.wiki_dir / rel_path

        if self.config.delete_mode == "log":
            logger.warning(
                f"Wiki page deleted: {title} (local: {rel_path}). "
                f"Manual action required to delete local file."
            )
            self.state.remove_file_state(rel_path)
        elif self.config.delete_mode == "sync":
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted local file: {file_path}")
            self.state.remove_file_state(rel_path)
        elif self.config.delete_mode == "archive":
            logger.warning(
                f"Archive mode not implemented. "
                f"Wiki page deleted: {title} (local: {rel_path})"
            )
            self.state.remove_file_state(rel_path)

    def _refresh_wiki_caches(self) -> None:
        """Refresh wiki caches after page changes (delete/create).

        1. Triggers immediate site stats refresh via HTTP endpoint
        2. Purges the main page parser cache
        """
        import hashlib
        import requests

        # 1. Trigger immediate stats refresh via HTTP endpoint
        try:
            token = hashlib.md5(
                f"{self.config.mw_user}{self.config.mw_pass}".encode()
            ).hexdigest()

            stats_url = f"{self.config.wiki_url}/refresh-stats.php"
            response = requests.get(
                stats_url,
                params={"token": token},
                timeout=30,
            )
            if response.ok:
                logger.info("Site stats refreshed successfully")
            else:
                logger.warning(f"Stats refresh returned: {response.status_code}")
        except Exception as e:
            logger.warning(f"Failed to refresh stats: {e}")

        # 2. Purge the main page parser cache
        try:
            self.mw.purge_page("Main_Page")
            logger.debug("Purged Main_Page cache")
        except Exception as e:
            logger.warning(f"Failed to purge Main_Page: {e}")

    def initial_sync(
        self,
        mode: str = "detect_only",
    ) -> dict:
        """
        Perform initial synchronization.

        Args:
            mode: "detect_only" - just build state without overwriting
                  "wiki_wins" - wiki overwrites local
                  "local_wins" - local overwrites wiki

        Returns:
            Dict with counts: synced_local, synced_wiki, conflicts, errors
        """
        stats = {
            "synced_local": 0,
            "synced_wiki": 0,
            "conflicts": 0,
            "errors": 0,
        }

        # Scan local files
        for folder in ["heuristics", "workflows", "principles",
                       "implementations", "environments"]:
            folder_path = self.config.wiki_dir / folder
            if not folder_path.exists():
                continue

            for file_path in folder_path.glob("*.md"):
                title = path_to_title(self.config.wiki_dir, file_path)
                if not title:
                    continue

                try:
                    local_content = normalize_text(
                        file_path.read_text(encoding="utf-8")
                    )
                except Exception as e:
                    logger.error(f"Failed to read {file_path}: {e}")
                    stats["errors"] += 1
                    continue

                if not local_content.strip():
                    continue

                local_hash = compute_content_hash(local_content)
                rel_path = str(file_path.relative_to(self.config.wiki_dir))

                # Get wiki state
                wiki_info = self.get_wiki_page_info(title)

                if mode == "detect_only":
                    # Just record current state
                    if wiki_info:
                        self.state.update_file_state(
                            rel_path=rel_path,
                            content_hash=local_hash,
                            wiki_title=title,
                            wiki_revid=wiki_info["revid"],
                        )
                    else:
                        # Page doesn't exist in wiki yet
                        self.state.update_file_state(
                            rel_path=rel_path,
                            content_hash=local_hash,
                            wiki_title=title,
                            wiki_revid=0,
                        )
                    stats["synced_local"] += 1

                elif mode == "wiki_wins":
                    if wiki_info:
                        # Overwrite local with wiki
                        file_path.write_text(wiki_info["content"], encoding="utf-8")
                        self.state.update_file_state(
                            rel_path=rel_path,
                            content_hash=compute_content_hash(wiki_info["content"]),
                            wiki_title=title,
                            wiki_revid=wiki_info["revid"],
                        )
                        stats["synced_wiki"] += 1
                    else:
                        # No wiki page - push local
                        self.sync_local_to_wiki(file_path)
                        stats["synced_local"] += 1

                elif mode == "local_wins":
                    # Push local to wiki
                    if self.sync_local_to_wiki(file_path):
                        stats["synced_local"] += 1
                    else:
                        stats["errors"] += 1

        self.state.save()
        return stats
