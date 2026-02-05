"""
Path <-> title transformations for wiki sync.

Handles conversion between local file paths and MediaWiki page titles.
"""

from pathlib import Path
from typing import Optional, Tuple
import unicodedata

# Map folder names to MediaWiki namespaces (from import_wiki_pages.py)
FOLDER_TO_NAMESPACE = {
    "heuristics": "Heuristic",
    "workflows": "Workflow",
    "principles": "Principle",
    "implementations": "Implementation",
    "environments": "Environment",
}

# Reverse mapping: namespace -> folder
NAMESPACE_TO_FOLDER = {v: k for k, v in FOLDER_TO_NAMESPACE.items()}

# Folders to skip during sync
SKIP_FOLDERS = {"_staging", "_reports", "_files", "_conflicts"}


def normalize_text(text: str) -> str:
    """Normalize unicode to NFC for consistent comparison."""
    return unicodedata.normalize("NFC", text)


def path_to_title(wiki_dir: Path, file_path: Path) -> Optional[str]:
    """
    Convert a local file path to a MediaWiki page title.

    Args:
        wiki_dir: Base wiki directory (e.g., /wikis)
        file_path: Full path to the .md file

    Returns:
        MediaWiki page title (e.g., "Heuristic:My_Page") or None if invalid
    """
    try:
        rel_path = file_path.relative_to(wiki_dir)
    except ValueError:
        return None

    parts = rel_path.parts
    if len(parts) < 1:
        return None

    folder_name = parts[0]

    # Skip special folders
    if folder_name in SKIP_FOLDERS or folder_name.startswith("_"):
        return None

    # Get filename without extension
    filename = file_path.name
    if filename.endswith(".md"):
        name = filename[:-3]
    elif filename.endswith(".mediawiki"):
        name = filename[:-10]
    else:
        return None

    # Replace spaces with underscores
    name = name.replace(" ", "_")

    # Get namespace from folder
    namespace = FOLDER_TO_NAMESPACE.get(folder_name, "")

    if namespace:
        return f"{namespace}:{name}"
    else:
        # Unsupported folder - skip
        return None


def title_to_path(wiki_dir: Path, title: str) -> Optional[Path]:
    """
    Convert a MediaWiki page title to a local file path.

    Args:
        wiki_dir: Base wiki directory (e.g., /wikis)
        title: MediaWiki page title (e.g., "Heuristic:My_Page")

    Returns:
        Path to local .md file or None if title is not in synced namespaces
    """
    # Parse namespace:name format
    if ":" in title:
        namespace, name = title.split(":", 1)
    else:
        # No namespace - not in our synced set
        return None

    # Get folder from namespace
    folder = NAMESPACE_TO_FOLDER.get(namespace)
    if not folder:
        return None

    folder_path = wiki_dir / folder

    # MediaWiki treats spaces and underscores as equivalent
    # Check for existing file with either convention
    name_with_spaces = name.replace("_", " ")
    name_with_underscores = name.replace(" ", "_")

    # Check for existing files (prefer underscore version to match original convention)
    path_underscores = folder_path / f"{name_with_underscores}.md"
    path_spaces = folder_path / f"{name_with_spaces}.md"

    if path_underscores.exists():
        return path_underscores
    elif path_spaces.exists():
        return path_spaces
    else:
        # New file - use underscore convention
        return path_underscores


def parse_namespace(title: str) -> Tuple[Optional[str], str]:
    """
    Parse a MediaWiki title into namespace and name components.

    Args:
        title: MediaWiki page title

    Returns:
        Tuple of (namespace, name). namespace is None for main namespace.
    """
    if ":" in title:
        namespace, name = title.split(":", 1)
        return namespace, name
    return None, title


def get_synced_namespaces() -> list[str]:
    """Return list of namespace names that are synced."""
    return list(NAMESPACE_TO_FOLDER.keys())


def is_synced_title(title: str) -> bool:
    """Check if a title belongs to a synced namespace."""
    namespace, _ = parse_namespace(title)
    return namespace in NAMESPACE_TO_FOLDER


def is_synced_path(wiki_dir: Path, file_path: Path) -> bool:
    """Check if a file path belongs to a synced folder."""
    try:
        rel_path = file_path.relative_to(wiki_dir)
    except ValueError:
        return False

    if len(rel_path.parts) < 1:
        return False

    folder = rel_path.parts[0]
    if folder in SKIP_FOLDERS or folder.startswith("_"):
        return False

    return folder in FOLDER_TO_NAMESPACE
