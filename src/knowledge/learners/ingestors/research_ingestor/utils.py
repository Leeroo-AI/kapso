# Research Ingestor Utilities
#
# Shared utility functions for research ingestors.
# Includes wiki structure loading, slugification, and path helpers.

import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def slugify(text: str, max_len: int = 60) -> str:
    """
    Create a filesystem-safe slug from text.
    
    Follows WikiMedia naming conventions:
    - Alphanumeric and underscores only
    - First character capitalized
    - No consecutive underscores
    
    Args:
        text: Raw text to slugify
        max_len: Maximum length of slug
        
    Returns:
        Filesystem-safe slug string
    """
    if not text:
        return "Unknown"
    
    # Replace non-alphanumeric with underscores
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip())
    
    # Collapse multiple underscores
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    
    # Strip leading/trailing underscores
    cleaned = cleaned.strip("_")
    
    # Capitalize first letter
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    
    return (cleaned or "Unknown")[:max_len]


def sanitize_wiki_title(text: str) -> str:
    """
    Sanitize a string for WikiMedia page title compliance.
    
    WikiMedia Naming Rules:
    - First character is auto-capitalized by the system
    - Underscores only as word separators (no hyphens, no spaces)
    - Forbidden characters: # < > [ ] { } | + : /
    - Alphanumeric and underscores only
    
    Args:
        text: Raw string to sanitize
        
    Returns:
        WikiMedia-compliant title string with first letter capitalized
    """
    if not text:
        return "X"
    
    # Forbidden WikiMedia characters: # < > [ ] { } | + : /
    # Also replace hyphens and spaces with underscores
    result = []
    for ch in text:
        if ch.isalnum():
            result.append(ch)
        elif ch == "_":
            result.append("_")
        else:
            # Replace forbidden chars, hyphens, spaces with underscore
            result.append("_")
    
    # Join and collapse multiple consecutive underscores
    sanitized = "".join(result)
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    
    # Strip leading/trailing underscores
    sanitized = sanitized.strip("_")
    
    # Capitalize first letter (WikiMedia convention)
    if sanitized:
        sanitized = sanitized[0].upper() + sanitized[1:]
    
    return sanitized or "X"


def get_wiki_structure_dir() -> Path:
    """
    Get the path to the wiki_structure directory.
    
    Returns:
        Path to src/knowledge/wiki_structure/
    """
    # Navigate from this file to wiki_structure
    # This file: src/knowledge/learners/ingestors/research_ingestor/utils.py
    # Target: src/knowledge/wiki_structure/
    return Path(__file__).parents[3] / "wiki_structure"


def load_wiki_structure(page_type: str) -> str:
    """
    Load wiki structure definitions for a page type.
    
    Loads and combines:
    - page_definition.md: Core definition and graph connectivity
    - sections_definition.md: Detailed section-by-section guide
    
    Args:
        page_type: Page type name (workflow, principle, implementation, 
                   environment, heuristic)
        
    Returns:
        Combined content of page_definition.md + sections_definition.md
        
    Raises:
        FileNotFoundError: If wiki structure directory doesn't exist
    """
    wiki_structure_dir = get_wiki_structure_dir()
    type_dir = wiki_structure_dir / f"{page_type.lower()}_page"
    
    if not type_dir.exists():
        raise FileNotFoundError(f"Wiki structure not found for type: {page_type}")
    
    content = f"# {page_type.title()} Page Structure\n\n"
    
    # Load page definition
    page_def = type_dir / "page_definition.md"
    if page_def.exists():
        content += "## Page Definition\n\n"
        content += page_def.read_text(encoding="utf-8") + "\n\n"
    
    # Load sections definition
    sections_def = type_dir / "sections_definition.md"
    if sections_def.exists():
        content += "## Sections Guide\n\n"
        content += sections_def.read_text(encoding="utf-8")
    
    return content


def load_all_wiki_structures() -> str:
    """
    Load wiki structure definitions for all page types.
    
    Returns:
        Combined content for all 4 page types (Principle, Implementation,
        Environment, Heuristic). Workflow is excluded as research outputs
        don't create workflows.
    """
    # Research outputs can create these page types
    page_types = ["principle", "implementation", "environment", "heuristic"]
    
    content = "# Wiki Structure Definitions\n\n"
    content += "The following defines the structure for each page type in the knowledge graph.\n"
    content += "Use these definitions to decide what page types to create and how to structure them.\n\n"
    
    for page_type in page_types:
        content += f"\n{'='*60}\n"
        try:
            content += load_wiki_structure(page_type)
        except FileNotFoundError:
            content += f"(No structure defined for {page_type})\n"
    
    return content


def load_page_connections() -> str:
    """
    Load the page_connections.md file that defines graph relationships.
    
    Returns:
        Content of page_connections.md
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    wiki_structure_dir = get_wiki_structure_dir()
    connections_file = wiki_structure_dir / "page_connections.md"
    
    if not connections_file.exists():
        raise FileNotFoundError(f"Page connections file not found: {connections_file}")
    
    return connections_file.read_text(encoding="utf-8")


def get_research_namespace() -> str:
    """
    Get the namespace prefix for research-generated pages.
    
    All pages created by research ingestors use this prefix
    to distinguish them from repo-extracted pages.
    
    Returns:
        "Research_Web" - indicates web research origin
    """
    return "Research_Web"


def build_page_filename(page_type: str, slug: str) -> str:
    """
    Build a WikiMedia-compliant filename for a page.
    
    Args:
        page_type: Type of page (principle, implementation, etc.)
        slug: Slugified page name (should be descriptive of content)
        
    Returns:
        Filename like "QLoRA_Best_Practices.md" (no namespace prefix)
    """
    sanitized_slug = sanitize_wiki_title(slug)
    return f"{sanitized_slug}.md"


def get_page_subdirectory(page_type: str) -> str:
    """
    Get the subdirectory name for a page type.
    
    Args:
        page_type: Type of page
        
    Returns:
        Subdirectory name (e.g., "principles", "implementations")
    """
    type_to_dir = {
        "principle": "principles",
        "implementation": "implementations",
        "environment": "environments",
        "heuristic": "heuristics",
        "workflow": "workflows",
    }
    return type_to_dir.get(page_type.lower(), page_type.lower() + "s")
