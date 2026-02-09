"""
Configuration for the wiki sync service using Pydantic settings.
"""

from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings


class SyncConfig(BaseSettings):
    """Configuration for bidirectional wiki sync service."""

    # MediaWiki connection
    wiki_url: str = "http://localhost:8090"
    mw_user: str = "agent"
    mw_pass: str = ""

    # Local paths
    wiki_dir: Path = Path("/wikis")
    sync_state: Path = Path("/state/sync.json")

    # Timing
    poll_interval: int = 30  # seconds between RC polls
    debounce_ms: int = 500  # file watcher debounce delay
    refresh_delay: int = 15  # seconds of quiet before Cargo/cache refresh

    # Behavior
    conflict_mode: Literal["file", "wiki_wins", "local_wins"] = "file"
    delete_mode: Literal["log", "sync", "archive"] = "log"

    # Rate limiting
    edit_rate_limit: float = 1.0  # max edits per second to wiki

    model_config = {
        "env_prefix": "",
        "env_file": ".env",
        "extra": "ignore",
    }

    @property
    def conflicts_dir(self) -> Path:
        """Directory for conflict files."""
        return self.wiki_dir / "_conflicts"
