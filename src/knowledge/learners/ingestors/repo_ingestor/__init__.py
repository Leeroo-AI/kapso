# Phased Repository Ingestor
#
# Extracts knowledge from Git repositories using a two-branch pipeline.
# The agent directly writes wiki pages following wiki_structure definitions.
#
# Branch 1: Workflow-Based Extraction
# 1. Anchoring - Find workflows from README/examples, write Workflow pages
# 2. Excavation - Trace imports to source code, write Implementation pages
# 3. Synthesis - Name theoretical concepts, write Principle pages
# 4. Enrichment - Mine constraints/tips, write Environment/Heuristic pages
# 5. Audit - Validate graph integrity, fix broken links
#
# Branch 2: Orphan Mining (runs after Branch 1)
# 6. Orphan Mining - Find uncaptured code, create nodes with polymorphism check
# 7. Orphan Audit - Validate orphan nodes, check for hidden workflows
#
# The final graph is the union of both branches.
#
# Usage:
#     from src.knowledge.learners.ingestors import RepoIngestor
#     
#     ingestor = RepoIngestor()
#     pages = ingestor.ingest(Source.Repo("https://github.com/user/repo"))

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.knowledge.learners.ingestors.base import Ingestor
from src.knowledge.learners.ingestors.factory import register_ingestor
from src.knowledge.search.base import WikiPage, DEFAULT_WIKI_DIR

from src.knowledge.learners.ingestors.repo_ingestor.utils import (
    clone_repo,
    cleanup_repo,
    load_wiki_structure,
    get_repo_name_from_url,
)

logger = logging.getLogger(__name__)

# Path to prompt templates
PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_prompt(name: str) -> str:
    """Load a prompt template from the prompts directory."""
    prompt_file = PROMPTS_DIR / f"{name}.md"
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt template not found: {prompt_file}")
    return prompt_file.read_text(encoding="utf-8")


@register_ingestor("repo")
class RepoIngestor(Ingestor):
    """
    Phased repository knowledge extraction with two-branch pipeline.
    
    The agent directly writes wiki pages following wiki_structure definitions.
    No intermediate dataclasses - clean and simple.
    
    The agent is given:
    - Read access to the cloned repository
    - Wiki structure definitions (page_definition.md, sections_definition.md)
    - Write access to wiki output directory
    
    Branch 1: Workflow-Based Extraction
    1. Anchoring: Find workflows → write Workflow pages
    2. Excavation: Trace imports → write Implementation pages
    3. Synthesis: Name principles → write Principle pages
    4. Enrichment: Mine constraints → write Environment/Heuristic pages
    5. Audit: Validate pages → fix broken links
    
    Branch 2: Orphan Mining
    6. Orphan Mining: Find uncaptured code → create nodes with polymorphism check
    7. Orphan Audit: Validate orphans → check for hidden workflows, dead code
    
    The final graph is the union of both branches.
    
    Example:
        ingestor = RepoIngestor()
        pages = ingestor.ingest(Source.Repo("https://github.com/unslothai/unsloth"))
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the phased repo ingestor.
        
        Args:
            params: Optional parameters:
                - timeout: Claude Code timeout in seconds (default: 1800)
                - cleanup: Whether to cleanup cloned repos (default: True)
                - wiki_dir: Output directory for wiki pages (default: data/wikis)
        """
        super().__init__(params)
        self._timeout = self.params.get("timeout", 1800)  # 30 minutes default
        self._cleanup = self.params.get("cleanup", True)
        self._wiki_dir = Path(self.params.get("wiki_dir", DEFAULT_WIKI_DIR))
        self._agent = None
        self._last_repo_path: Optional[Path] = None
    
    @property
    def source_type(self) -> str:
        """Return the source type this ingestor handles."""
        return "repo"
    
    def _initialize_agent(self, workspace: str) -> None:
        """
        Initialize Claude Code agent with read + write tools.
        
        Args:
            workspace: Path to the cloned repository
        """
        try:
            from src.execution.coding_agents.factory import CodingAgentFactory
            
            # Build config for Claude Code with read + write tools
            config = CodingAgentFactory.build_config(
                agent_type="claude_code",
                agent_specific={
                    # Read for repo exploration, Write for wiki pages, Bash for file ops
                    "allowed_tools": ["Read", "Write", "Bash"],
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
    
    def _ensure_wiki_directories(self) -> None:
        """Ensure wiki subdirectories exist."""
        for subdir in ["workflows", "principles", "implementations", "environments", "heuristics"]:
            (self._wiki_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    def _build_phase_prompt(
        self,
        phase: str,
        repo_name: str,
        repo_path: str,
    ) -> str:
        """
        Build prompt for a specific phase.
        
        Includes:
        - Phase-specific instructions from prompt template
        - Relevant wiki structure definitions
        - Paths for input/output
        
        Args:
            phase: Phase name (anchoring, excavation, synthesis, enrichment, audit)
            repo_name: Name of the repository
            repo_path: Path to the cloned repository
            
        Returns:
            Complete prompt string
        """
        # Load base prompt template
        base_prompt = _load_prompt(phase)
        
        # Determine which wiki structures to include
        structures_needed = {
            # Branch 1: Workflow-based extraction
            "anchoring": ["workflow"],
            "excavation": ["implementation"],
            "synthesis": ["principle"],
            "enrichment": ["environment", "heuristic"],
            "audit": ["workflow", "principle", "implementation", "environment", "heuristic"],
            # Branch 2: Orphan mining
            "orphan_mining": ["implementation", "principle"],
            "orphan_audit": ["workflow", "implementation", "principle", "heuristic"],
        }
        
        # Load relevant wiki structure definitions
        structure_content = {}
        for page_type in structures_needed.get(phase, []):
            try:
                structure_content[f"{page_type}_structure"] = load_wiki_structure(page_type)
            except FileNotFoundError:
                structure_content[f"{page_type}_structure"] = f"(No structure defined for {page_type})"
        
        # Format the prompt with all variables
        format_vars = {
            "repo_name": repo_name,
            "repo_path": repo_path,
            "wiki_dir": str(self._wiki_dir),
            **structure_content,
        }
        
        try:
            return base_prompt.format(**format_vars)
        except KeyError as e:
            # If template has variables we don't have, just return with partial formatting
            logger.warning(f"Missing format variable in {phase} prompt: {e}")
            return base_prompt
    
    def _run_phase(self, phase: str, repo_name: str, repo_path: str) -> bool:
        """
        Run a single phase of the extraction pipeline.
        
        Args:
            phase: Phase name
            repo_name: Repository name
            repo_path: Path to cloned repository
            
        Returns:
            True if phase succeeded, False otherwise
        """
        logger.info(f"Running {phase} phase...")
        
        prompt = self._build_phase_prompt(phase, repo_name, repo_path)
        result = self._agent.generate_code(prompt)
        
        if not result.success:
            logger.error(f"{phase} phase failed: {result.error}")
            return False
        
        logger.info(f"{phase} phase complete")
        return True
    
    def _collect_written_pages(self, repo_name: str) -> List[WikiPage]:
        """
        Collect WikiPage objects from files written by agent.
        
        Scans wiki_dir for .md files and parses them.
        
        Args:
            repo_name: Repository name to filter pages
            
        Returns:
            List of WikiPage objects
        """
        try:
            from src.knowledge.search.kg_graph_search import parse_wiki_directory
            
            pages = parse_wiki_directory(self._wiki_dir)
            
            # Filter to pages from this repo (optional - may want all pages)
            # For now, return all pages in the wiki directory
            logger.info(f"Collected {len(pages)} pages from {self._wiki_dir}")
            return pages
            
        except ImportError:
            logger.warning("Could not import parse_wiki_directory, returning empty list")
            return []
        except Exception as e:
            logger.error(f"Failed to collect pages: {e}")
            return []
    
    def ingest(self, source: Any) -> List[WikiPage]:
        """
        Run the two-branch extraction pipeline (7 phases total).
        
        Branch 1 (Workflow-based): anchoring → excavation → synthesis → enrichment → audit
        Branch 2 (Orphan mining): orphan_mining → orphan_audit
        
        Args:
            source: Source.Repo object or dict with:
                - url: GitHub repository URL (required)
                - branch: Branch to clone (default: "main")
            
        Returns:
            List of WikiPage objects extracted from the repository
            (union of both branches)
            
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
            repo_name = get_repo_name_from_url(url)
            
            # Step 2: Ensure wiki directories exist
            self._ensure_wiki_directories()
            
            # Step 3: Initialize Claude Code agent
            self._initialize_agent(str(repo_path))
            
            # Step 4: Run Branch 1 - Workflow-based extraction
            logger.info("=" * 60)
            logger.info("BRANCH 1: Workflow-Based Extraction")
            logger.info("=" * 60)
            
            branch1_phases = ["anchoring", "excavation", "synthesis", "enrichment", "audit"]
            
            for phase in branch1_phases:
                success = self._run_phase(phase, repo_name, str(repo_path))
                if not success:
                    logger.warning(f"Phase {phase} failed, continuing to next phase...")
                    # Continue even if a phase fails - try to extract what we can
            
            # Step 5: Run Branch 2 - Orphan mining
            logger.info("=" * 60)
            logger.info("BRANCH 2: Orphan Mining")
            logger.info("=" * 60)
            
            branch2_phases = ["orphan_mining", "orphan_audit"]
            
            for phase in branch2_phases:
                success = self._run_phase(phase, repo_name, str(repo_path))
                if not success:
                    logger.warning(f"Phase {phase} failed, continuing to next phase...")
            
            # Step 6: Collect pages written by agent (union of both branches)
            pages = self._collect_written_pages(repo_name)
            
            logger.info(f"Extracted {len(pages)} pages from {url}")
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


# Alias for backward compatibility
PhasedRepoIngestor = RepoIngestor

