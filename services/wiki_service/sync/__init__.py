"""
Bidirectional Wiki Sync Service

Keeps local files in data/wikis/ synchronized with MediaWiki pages:
- Local -> Wiki: File changes detected via watchdog push to MediaWiki API
- Wiki -> Local: Page edits detected via RecentChanges polling pull to local files
"""

from .config import SyncConfig
from .state_manager import StateManager
from .sync_engine import SyncEngine

__all__ = ["SyncConfig", "StateManager", "SyncEngine"]
