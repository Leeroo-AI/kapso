"""
Wiki content reader service.

Reads wiki content from the filesystem (mounted data/wikis directory).
"""

import re
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from ..config import get_settings


# Valid namespaces mapping (directory name to display name)
NAMESPACE_DIRS: Dict[str, str] = {
    "principles": "Principle",
    "workflows": "Workflow",
    "implementations": "Implementation",
    "artifacts": "Artifact",
    "heuristics": "Heuristic",
    "environments": "Environment",
    "resources": "Resource",
}

# Namespace display name to ID mapping
NAMESPACE_IDS: Dict[str, int] = {
    "main": 0,
    "Principle": 3000,
    "Workflow": 3002,
    "Implementation": 3004,
    "Artifact": 3006,
    "Heuristic": 3008,
    "Environment": 3010,
    "Resource": 3012,
}

# For API responses
VALID_NAMESPACES: Dict[int, str] = {v: k for k, v in NAMESPACE_IDS.items()}

# Namespace name variations for filtering
NAMESPACE_NAME_TO_DIR: Dict[str, str] = {
    "main": "",
    "principle": "principles",
    "principles": "principles",
    "workflow": "workflows",
    "workflows": "workflows",
    "implementation": "implementations",
    "implementations": "implementations",
    "artifact": "artifacts",
    "artifacts": "artifacts",
    "heuristic": "heuristics",
    "heuristics": "heuristics",
    "environment": "environments",
    "environments": "environments",
    "resource": "resources",
    "resources": "resources",
}


@dataclass
class WikiPage:
    """Represents a wiki page."""

    title: str
    namespace: str
    namespace_id: int
    content: str


@dataclass
class PageInfo:
    """Basic page info without content."""

    title: str
    namespace: str
    namespace_id: int


class WikiReader:
    """
    Reads wiki content from the filesystem.

    Wiki files are organized as:
        data/wikis/<namespace_dir>/<Title>.md

    For example:
        data/wikis/implementations/Unslothai_Unsloth_FastLanguageModel.md
    """

    def __init__(self):
        self.settings = get_settings()
        self.wiki_path = Path(self.settings.wiki_data_path)

    def _sanitize_title(self, title: str) -> Optional[str]:
        """
        Sanitize a page title for filesystem access.

        Returns None if the title contains invalid characters.
        """
        # Whitelist: alphanumeric, underscores, hyphens, dots
        if not re.match(r"^[a-zA-Z0-9_\-\.]+$", title):
            return None

        # Block traversal patterns
        if ".." in title or "/" in title or "\\" in title:
            return None

        return title

    def _get_namespace_info(self, dir_name: str) -> tuple[str, int]:
        """Get namespace display name and ID from directory name."""
        display_name = NAMESPACE_DIRS.get(dir_name, "main")
        ns_id = NAMESPACE_IDS.get(display_name, 0)
        return display_name, ns_id

    def list_pages(self, namespace: Optional[str] = None) -> List[PageInfo]:
        """
        List all available wiki pages.

        Args:
            namespace: Optional namespace filter (e.g., "implementation")

        Returns:
            List of PageInfo objects
        """
        pages = []

        if not self.wiki_path.exists():
            return []

        # Determine which directories to scan
        if namespace:
            ns_lower = namespace.lower()
            target_dir = NAMESPACE_NAME_TO_DIR.get(ns_lower)
            if target_dir is None:
                return []  # Invalid namespace
            if target_dir:
                dirs_to_scan = [self.wiki_path / target_dir]
            else:
                # "main" namespace - would be root level files
                dirs_to_scan = [self.wiki_path]
        else:
            # Scan all namespace directories
            dirs_to_scan = [
                self.wiki_path / d for d in NAMESPACE_DIRS.keys()
                if (self.wiki_path / d).is_dir()
            ]

        for ns_dir in dirs_to_scan:
            if not ns_dir.exists() or not ns_dir.is_dir():
                continue

            dir_name = ns_dir.name
            display_name, ns_id = self._get_namespace_info(dir_name)

            # List all .md files
            for wiki_file in sorted(ns_dir.glob("*.md")):
                title = wiki_file.stem  # Remove .md extension

                pages.append(PageInfo(
                    title=title,
                    namespace=display_name,
                    namespace_id=ns_id
                ))

        return pages

    def get_page(self, namespace: str, title: str) -> Optional[WikiPage]:
        """
        Get a specific wiki page by namespace and title.

        Args:
            namespace: The namespace (e.g., "implementation", "workflow")
            title: The page title

        Returns:
            WikiPage if found, None otherwise
        """
        # Get directory for namespace
        ns_lower = namespace.lower()
        ns_dir = NAMESPACE_NAME_TO_DIR.get(ns_lower)
        if ns_dir is None:
            return None

        # Sanitize title
        safe_title = self._sanitize_title(title)
        if safe_title is None:
            return None

        # Build file path
        if ns_dir:
            file_path = self.wiki_path / ns_dir / f"{safe_title}.md"
        else:
            file_path = self.wiki_path / f"{safe_title}.md"

        # Verify path stays within wiki directory
        try:
            resolved = file_path.resolve()
            wiki_resolved = self.wiki_path.resolve()
            if not str(resolved).startswith(str(wiki_resolved)):
                return None
        except (OSError, ValueError):
            return None

        if not file_path.exists():
            return None

        # Read content
        try:
            content = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None

        display_name, ns_id = self._get_namespace_info(ns_dir) if ns_dir else ("main", 0)

        return WikiPage(
            title=title,
            namespace=display_name,
            namespace_id=ns_id,
            content=content
        )

    def export_all(self) -> List[Dict[str, Any]]:
        """
        Export all wiki content as a list of dictionaries.

        Returns:
            List of page dictionaries with title, namespace, content
        """
        pages = self.list_pages()
        result = []

        for page_info in pages:
            page = self.get_page(page_info.namespace, page_info.title)
            if page:
                result.append({
                    "title": page.title,
                    "namespace": page.namespace,
                    "namespace_id": page.namespace_id,
                    "content": page.content
                })

        return result


# Global reader instance
_wiki_reader: Optional[WikiReader] = None


def get_wiki_reader() -> WikiReader:
    """Get or create the global wiki reader."""
    global _wiki_reader
    if _wiki_reader is None:
        _wiki_reader = WikiReader()
    return _wiki_reader
