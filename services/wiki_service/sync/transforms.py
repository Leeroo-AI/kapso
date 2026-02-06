"""
Path <-> title transformations for wiki sync.

Handles conversion between local file paths and MediaWiki page titles,
and content transformations for pushing local files to MediaWiki.
"""

import re
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


# =============================================================================
# Content Transformations (local .md -> MediaWiki-ready content)
# =============================================================================
#
# These mirror the transforms applied by import_wikis.sh (lines 122-144)
# so that the sync engine produces identical wiki content.


def strip_h1_heading(content: str) -> str:
    """
    Strip redundant first-line heading from page content.

    MediaWiki already displays the page title in its header, so a
    duplicate H1 heading at the top of the body is unnecessary.
    Markdown H1 (``# Title``) renders as a numbered list item in
    MediaWiki syntax, which causes the "1. Workflow: ..." bug.

    Matches import_wikis.sh behaviour (lines 127-129).
    """
    if not content:
        return content

    lines = content.split("\n")
    first_line = lines[0]

    # Check for Markdown H1 ("# ...") or MediaWiki H1 ("= ... =")
    if first_line.startswith("# ") or first_line.startswith("= "):
        # Remove the heading line
        lines = lines[1:]
        # Also strip any leading blank lines that followed the heading
        while lines and not lines[0].strip():
            lines = lines[1:]
        return "\n".join(lines)

    return content


# Regex: [[source::Type|DisplayText|URL]] -> [URL DisplayText]
# Groups: (1) Type (ignored), (2) DisplayText, (3) URL
_SOURCE_LINK_RE = re.compile(
    r"\[\[source::[^|]+\|([^|]+)\|(https?:[^\]]+)\]\]"
)


def transform_source_links(content: str) -> str:
    """
    Convert semantic source annotations to clickable MediaWiki external links.

    ``[[source::Repo|OpenClaw|https://github.com/...]]``
    becomes ``[https://github.com/... OpenClaw]``

    The original triple-pipe syntax is preserved in the local .md files
    for the internal knowledge-graph parser (kg_graph_search.py).
    This function only transforms content destined for MediaWiki rendering.

    Matches import_wikis.sh behaviour (line 135).
    """
    return _SOURCE_LINK_RE.sub(r"[\2 \1]", content)


def add_page_metadata(content: str, namespace: str, page_name: str) -> str:
    """
    Prepend Cargo template and append Category tag.

    These are needed for Cargo queries (PageInfo table) and MediaWiki
    category browsing. Skips if already present to avoid duplication
    during repeated syncs.

    Matches import_wikis.sh behaviour (lines 138-144).

    Args:
        content: Page content (after other transforms)
        namespace: MediaWiki namespace (e.g. "Workflow")
        page_name: Page name without namespace (e.g. "Openclaw_Openclaw_QLoRA")
    """
    # Pluralize namespace for category name (Workflow -> Workflows)
    category = f"{namespace}s"

    # Prepend PageInfo template if not already present
    pageinfo_tag = "{{PageInfo|"
    if pageinfo_tag not in content:
        content = f"{{{{PageInfo|type={namespace}|title={page_name}}}}}\n{content}"

    # Append category tag if not already present
    category_tag = f"[[Category:{category}]]"
    if category_tag not in content:
        content = f"{content}\n\n{category_tag}"

    return content


def prepare_content_for_wiki(
    content: str,
    namespace: Optional[str] = None,
    page_name: Optional[str] = None,
) -> str:
    """
    Apply all transforms to prepare local .md content for MediaWiki.

    This is the single entry point used by sync_engine and import tools.
    Mirrors the full transform pipeline from import_wikis.sh.

    Steps:
        1. Strip redundant H1 heading
        2. Convert [[source::...]] to [URL Label] external links
        3. Add PageInfo template and Category tag (if namespace provided)

    Args:
        content: Raw local file content
        namespace: MediaWiki namespace (e.g. "Workflow"), or None to skip metadata
        page_name: Page name without namespace, or None to skip metadata
    """
    content = strip_h1_heading(content)
    content = transform_source_links(content)

    # Add metadata only when namespace info is available
    if namespace and page_name:
        content = add_page_metadata(content, namespace, page_name)

    return content
