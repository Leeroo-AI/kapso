# Repository Ingestor Utilities
#
# Helper functions for the phased repo ingestor:
# - clone_repo: Clone a git repository to temp directory
# - cleanup_repo: Remove cloned repository
# - load_wiki_structure: Load wiki page definitions from wiki_structure/

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

def clone_repo(url: str, branch: str = "main") -> Path:
    """
    Clone a Git repository to a temporary directory.
    
    Uses shallow clone (depth=1) for efficiency.
    
    Args:
        url: GitHub repository URL
        branch: Branch to clone (default: main)
        
    Returns:
        Path to the cloned repository
        
    Raises:
        RuntimeError: If git clone fails
    """
    # Create temp directory with recognizable prefix
    temp_dir = tempfile.mkdtemp(prefix="kapso_repo_")
    
    logger.info(f"Cloning {url} (branch: {branch}) to {temp_dir}")
    
    try:
        result = subprocess.run(
            ["git", "clone", "--depth", "1", "-b", branch, url, temp_dir],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout for large repos
        )
        
        if result.returncode != 0:
            # Try without branch specification (might be 'master' instead of 'main')
            logger.info(f"Branch '{branch}' failed, trying default branch...")
            
            # Clean up failed attempt
            shutil.rmtree(temp_dir, ignore_errors=True)
            temp_dir = tempfile.mkdtemp(prefix="kapso_repo_")
            
            result = subprocess.run(
                ["git", "clone", "--depth", "1", url, temp_dir],
                capture_output=True,
                text=True,
                timeout=300,
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Git clone failed: {result.stderr}")
        
        logger.info(f"Successfully cloned repository to {temp_dir}")
        return Path(temp_dir)
        
    except subprocess.TimeoutExpired:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError(f"Git clone timed out for {url}")
    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError(f"Failed to clone repository: {e}")


def cleanup_repo(repo_path: Path) -> None:
    """
    Remove a cloned repository directory.
    
    Args:
        repo_path: Path to the cloned repository
    """
    if repo_path and repo_path.exists():
        logger.info(f"Cleaning up {repo_path}")
        shutil.rmtree(repo_path, ignore_errors=True)


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
    # Path to wiki structure definitions (relative to this file's location)
    wiki_structure_dir = Path(__file__).parents[3] / "wiki_structure"
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
        Combined content for all 5 page types
    """
    page_types = ["workflow", "principle", "implementation", "environment", "heuristic"]
    
    content = "# Wiki Structure Definitions\n\n"
    content += "The following defines the structure for each page type in the knowledge graph.\n\n"
    
    for page_type in page_types:
        content += f"\n{'='*60}\n"
        try:
            content += load_wiki_structure(page_type)
        except FileNotFoundError:
            content += f"(No structure defined for {page_type})\n"
    
    return content


def get_repo_name_from_url(url: str) -> str:
    """
    Extract repository name from GitHub URL.
    
    Args:
        url: GitHub repository URL (e.g., https://github.com/user/repo)
        
    Returns:
        Repository name (e.g., "repo")
    """
    # Remove trailing slashes and .git suffix
    url = url.rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]
    
    return url.split("/")[-1]


def sanitize_wiki_title(s: str) -> str:
    """
    Sanitize a string for WikiMedia page title compliance.
    
    WikiMedia Naming Rules:
    - First character is auto-capitalized by the system
    - Underscores only as word separators (no hyphens, no spaces)
    - Forbidden characters: # < > [ ] { } | + : /
    - Alphanumeric and underscores only
    
    Args:
        s: Raw string to sanitize
        
    Returns:
        WikiMedia-compliant title string with first letter capitalized
    """
    if not s:
        return "X"
    
    # Forbidden WikiMedia characters: # < > [ ] { } | + : /
    # Also replace hyphens and spaces with underscores
    result = []
    for ch in s:
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


def get_repo_namespace_from_url(url: str) -> str:
    """
    Extract a stable, collision-resistant repo namespace from a git URL.
    
    Follows WikiMedia naming conventions:
    - First character uppercase
    - Underscores only (no hyphens)
    - No forbidden characters: # < > [ ] { } | + : /
    
    This is used to:
    - Prefix wiki filenames (prevents cross-repo collisions in a shared wiki_dir)
    - Scope phases to only this repo's pages during Audit / Orphan passes
    
    Supported inputs:
    - https://github.com/owner/repo
    - https://github.com/owner/repo.git
    - git@github.com:owner/repo.git
    
    Returns:
        A WikiMedia-compliant namespace string like: "Owner_Repo"
    """
    raw = (url or "").strip()
    if not raw:
        return "Unknown_Repo"
    
    # Normalize .git suffix and trailing slash
    raw = raw.rstrip("/")
    if raw.endswith(".git"):
        raw = raw[:-4]
    
    owner = None
    repo = None
    
    # SSH style: git@github.com:owner/repo
    if raw.startswith("git@"):
        # Split on ":" then "/"
        try:
            path = raw.split(":", 1)[1]
            parts = [p for p in path.split("/") if p]
            if len(parts) >= 2:
                owner, repo = parts[0], parts[1]
        except Exception:
            owner, repo = None, None
    else:
        # HTTPS or other URL style
        parsed = urlparse(raw)
        # If it wasn't a real URL, treat it like a path
        path = parsed.path if parsed.scheme else raw
        parts = [p for p in path.split("/") if p]
        if len(parts) >= 2:
            owner, repo = parts[-2], parts[-1]
        elif len(parts) == 1:
            repo = parts[0]
    
    # Fallbacks
    if not repo:
        repo = "Repo"
    if not owner:
        owner = "Owner"
    
    # Apply WikiMedia-compliant sanitization
    return f"{sanitize_wiki_title(owner)}_{sanitize_wiki_title(repo)}"

