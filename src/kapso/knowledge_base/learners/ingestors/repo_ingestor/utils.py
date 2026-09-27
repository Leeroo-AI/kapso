# Repository Ingestor Utilities
#
# Helper functions for the phased repo ingestor:
# - clone_repo: Clone a git repository into a run's staging directory
# - checked_out_branch: The branch a clone is on
# - load_wiki_structure: Load wiki page definitions from wiki_structure/

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Name of the clone inside a run's staging directory. A page that cites a
# path through it points at a directory that will not outlive the run, which
# the validator rejects.
CLONE_DIR_NAME = "kapso_repo_clone"


def clone_repo(url: str, branch: Optional[str], dest: Path) -> None:
    """
    Shallow-clone a repository into `dest`.
    
    Args:
        url: Repository URL
        branch: Branch to check out; None takes the repository's default branch
        dest: Directory to clone into (replaced if a previous clone was cut short)
        
    Raises:
        RuntimeError: If git cannot clone the repository or the branch. Git is
            told not to prompt for credentials, so a private or missing
            repository fails here instead of waiting on a terminal.
    """
    if dest.exists():
        shutil.rmtree(dest)
    command = ["git", "clone", "--depth", "1"]
    if branch:
        command += ["--branch", branch]
    command += [url, str(dest)]
    logger.info(f"Cloning {url} (branch: {branch or 'default'}) to {dest}")
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    if result.returncode != 0:
        shutil.rmtree(dest, ignore_errors=True)
        wanted = f" (branch {branch})" if branch else ""
        raise RuntimeError(f"git clone failed for {url}{wanted}: {result.stderr.strip()}")
    logger.info(f"Cloned {url} to {dest}")


def checked_out_branch(repo_path: Path) -> str:
    """The branch a clone has checked out, as named by the repository."""
    result = subprocess.run(
        ["git", "-C", str(repo_path), "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"{repo_path} is not a git clone: {result.stderr.strip()}")
    return result.stdout.strip()


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

