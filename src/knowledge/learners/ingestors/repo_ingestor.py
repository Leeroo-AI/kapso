# Repository Ingestor
#
# Extracts knowledge from Git repositories using Claude Code agent.
# Clones the repo, explores it with Claude Code, and returns proposed WikiPages.
#
# Part of Stage 1 of the knowledge learning pipeline.

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.knowledge.learners.ingestors.base import Ingestor
from src.knowledge.learners.ingestors.factory import register_ingestor
from src.knowledge.search.base import WikiPage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Path to extraction instructions prompt
EXTRACTION_PROMPT_PATH = Path(__file__).parent / "prompts" / "repo_kg_extraction_instructions.md"


# =============================================================================
# Helper Functions
# =============================================================================

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
    temp_dir = tempfile.mkdtemp(prefix="praxium_repo_")
    
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


def load_extraction_prompt() -> str:
    """
    Load the extraction instructions prompt from file.
    
    Returns:
        The extraction prompt text
        
    Raises:
        FileNotFoundError: If prompt file doesn't exist
    """
    if not EXTRACTION_PROMPT_PATH.exists():
        raise FileNotFoundError(
            f"Extraction prompt not found: {EXTRACTION_PROMPT_PATH}"
        )
    
    return EXTRACTION_PROMPT_PATH.read_text(encoding="utf-8")


def parse_extraction_output(output: str) -> List[WikiPage]:
    """
    Parse Claude Code agent output into WikiPage objects.
    
    Extracts JSON from the agent's response and converts to WikiPage list.
    
    Args:
        output: Raw output from Claude Code agent
        
    Returns:
        List of proposed WikiPage objects
    """
    json_str = None
    
    # Try to find JSON in markdown code blocks
    # Use non-greedy match for the opening, greedy for content
    json_match = re.search(r'```(?:json)?\s*\n(\{.*\})\s*\n```', output, re.DOTALL)
    
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find raw JSON object containing proposed_pages
        # Look for the outermost JSON object
        start_idx = output.find('{"')
        if start_idx == -1:
            start_idx = output.find('{\n')
        
        if start_idx != -1:
            # Find the matching closing brace
            brace_count = 0
            end_idx = start_idx
            for i, char in enumerate(output[start_idx:], start_idx):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break
            
            if end_idx > start_idx:
                json_str = output[start_idx:end_idx]
    
    if not json_str:
        logger.warning("Could not find JSON in agent output")
        return []
    
    # Try to parse the JSON
    data = None
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.debug(f"Initial JSON parse failed: {e}")
        
        # Try to fix common issues
        try:
            import ast
            # Try using ast.literal_eval for Python-like dict syntax
            data = ast.literal_eval(json_str)
        except (ValueError, SyntaxError):
            pass
        
        if data is None:
            logger.error(f"Failed to parse JSON from agent output: {e}")
            logger.debug(f"JSON string (first 500 chars): {json_str[:500]}")
            return []
    
    # Convert to WikiPage objects
    pages = []
    proposed = data.get("proposed_pages", [])
    
    for page_data in proposed:
        try:
            # Construct page ID from type and title
            page_type = page_data.get("page_type", "")
            page_title = page_data.get("page_title", "")
            page_id = f"{page_type}/{page_title}"
            
            page = WikiPage(
                id=page_id,
                page_title=page_title,
                page_type=page_type,
                overview=page_data.get("overview", ""),
                content=page_data.get("content", ""),
                domains=page_data.get("domains", []),
                sources=page_data.get("sources", []),
                outgoing_links=page_data.get("outgoing_links", []),
            )
            pages.append(page)
            
        except Exception as e:
            logger.warning(f"Failed to parse page: {e}")
            continue
    
    logger.info(f"Parsed {len(pages)} proposed pages from agent output")
    return pages


# =============================================================================
# RepoIngestor Class
# =============================================================================

@register_ingestor("repo")
class RepoIngestor(Ingestor):
    """
    Extract knowledge from Git repositories using Claude Code agent.
    
    This ingestor clones a repository, runs a Claude Code agent to explore it,
    and returns proposed WikiPage objects representing the extracted knowledge.
    
    Input formats:
        Source.Repo("https://github.com/user/repo")
        {"url": "https://github.com/user/repo", "branch": "main"}
    
    Example:
        ingestor = RepoIngestor()
        pages = ingestor.ingest(Source.Repo("https://github.com/unslothai/unsloth"))
        
        # Or with dict:
        pages = ingestor.ingest({"url": "https://github.com/user/repo"})
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize RepoIngestor.
        
        Args:
            params: Optional parameters:
                - timeout: Claude Code timeout in seconds (default: 1800)
                - cleanup: Whether to cleanup cloned repos (default: True)
        """
        super().__init__(params)
        self._timeout = self.params.get("timeout", 1800)  # 30 minutes default
        self._cleanup = self.params.get("cleanup", True)
        self._agent = None
        self._last_repo_path: Optional[Path] = None
    
    @property
    def source_type(self) -> str:
        """Return the source type this ingestor handles."""
        return "repo"
    
    def _initialize_agent(self, workspace: str) -> None:
        """
        Initialize Claude Code agent for the workspace.
        
        Args:
            workspace: Path to the cloned repository
        """
        try:
            from src.execution.coding_agents.factory import CodingAgentFactory
            
            # Build config for Claude Code with read-only tools
            config = CodingAgentFactory.build_config(
                agent_type="claude_code",
                agent_specific={
                    "allowed_tools": ["Read", "Bash"],  # Read-only exploration
                    "timeout": self._timeout,
                    "planning_mode": True,
                }
            )
            
            self._agent = CodingAgentFactory.create(config)
            self._agent.initialize(workspace)
            logger.info(f"Initialized Claude Code agent for {workspace}")
            
        except ImportError as e:
            logger.error(f"Could not import CodingAgentFactory: {e}")
            raise RuntimeError(
                "Claude Code agent not available. "
                "Ensure coding_agents module is properly installed."
            )
        except Exception as e:
            logger.error(f"Failed to initialize Claude Code agent: {e}")
            raise
    
    def _normalize_source(self, source: Any) -> Dict[str, Any]:
        """
        Normalize source input to a dict.
        
        Args:
            source: Source.Repo object or dict
            
        Returns:
            Dict with url and branch keys
        """
        if hasattr(source, 'to_dict'):
            return source.to_dict()
        elif isinstance(source, dict):
            return source
        else:
            raise ValueError(f"Invalid source type: {type(source)}")
    
    def ingest(self, source: Any) -> List[WikiPage]:
        """
        Extract knowledge from a repository and return WikiPage objects.
        
        Args:
            source: Source.Repo object or dict with:
                - url: GitHub repository URL (required)
                - branch: Branch to clone (default: "main")
            
        Returns:
            List of proposed WikiPage objects
            
        Raises:
            ValueError: If url is not provided
            RuntimeError: If cloning or extraction fails
        """
        # Normalize source to dict
        source_data = self._normalize_source(source)
        
        url = source_data.get("url", "")
        branch = source_data.get("branch", "main")
        
        if not url:
            raise ValueError("Repository URL is required")
        
        repo_path = None
        
        try:
            # Step 1: Clone the repository
            repo_path = clone_repo(url, branch)
            self._last_repo_path = repo_path
            
            # Step 2: Initialize Claude Code agent
            self._initialize_agent(str(repo_path))
            
            # Step 3: Load extraction prompt
            extraction_prompt = load_extraction_prompt()
            
            # Step 4: Run the agent
            logger.info("Running Claude Code agent for knowledge extraction...")
            result = self._agent.generate_code(extraction_prompt)
            
            if not result.success:
                logger.error(f"Agent extraction failed: {result.error}")
                return []
            
            # Step 5: Parse the output into WikiPages
            pages = parse_extraction_output(result.output)
            
            # Add source URL to all pages
            for page in pages:
                # Ensure the repo URL is in sources
                has_repo_source = any(
                    s.get("type") == "Repo" and url in s.get("url", "")
                    for s in page.sources
                )
                if not has_repo_source:
                    page.sources.append({
                        "type": "Repo",
                        "title": url.split("/")[-1],  # Repo name from URL
                        "url": url,
                    })
            
            logger.info(f"Extracted {len(pages)} proposed pages from {url}")
            return pages
            
        finally:
            # Cleanup cloned repository
            if self._cleanup and repo_path:
                cleanup_repo(repo_path)
                self._last_repo_path = None
    
    def get_last_repo_path(self) -> Optional[Path]:
        """
        Get the path to the last cloned repository.
        
        Useful for debugging or if cleanup=False was set.
        
        Returns:
            Path to the cloned repo, or None if cleaned up
        """
        return self._last_repo_path


# =============================================================================
# CLI Test
# =============================================================================

if __name__ == "__main__":
    """Test the RepoIngestor with a sample repository."""
    import sys
    
    print("=" * 60)
    print("RepoIngestor Test")
    print("=" * 60)
    
    # Default test repo
    test_url = sys.argv[1] if len(sys.argv) > 1 else "https://github.com/unslothai/unsloth"
    
    print(f"\nTest repository: {test_url}")
    print("-" * 60)
    
    # Create ingestor with cleanup disabled for inspection
    ingestor = RepoIngestor(params={"cleanup": False})
    
    try:
        # Run extraction
        pages = ingestor.ingest({"url": test_url})
        
        print(f"\nExtracted {len(pages)} proposed pages:")
        print("-" * 60)
        
        for i, page in enumerate(pages):
            print(f"\n[{i+1}] {page.page_title} ({page.page_type})")
            print(f"    Overview: {page.overview[:100]}...")
            print(f"    Domains: {page.domains}")
            print(f"    Links: {len(page.outgoing_links)} outgoing")
        
        # Show repo path for inspection
        if ingestor.get_last_repo_path():
            print(f"\nCloned repo available at: {ingestor.get_last_repo_path()}")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Test complete!")

