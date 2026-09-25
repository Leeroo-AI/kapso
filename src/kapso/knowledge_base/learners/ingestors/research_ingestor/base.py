# Research Ingestor Base Class
#
# Base class for agentic research ingestors.
# Provides Claude Code agent initialization, wiki structure loading,
# three-phase pipeline execution, and page collection.
#
# Usage:
#     class IdeaIngestor(ResearchIngestorBase):
#         @property
#         def source_type(self) -> str:
#             return "idea"

import logging
import time
import uuid
from abc import abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from kapso.execution.coding_agents.factory import CodingAgentFactory
from kapso.knowledge_base.learners.defaults import learner_defaults
from kapso.knowledge_base.learners.ingestors.base import Ingestor
from kapso.knowledge_base.search.base import WikiPage, DEFAULT_WIKI_DIR
from kapso.knowledge_base.search.kg_graph_search import parse_wiki_directory

from kapso.knowledge_base.learners.ingestors.research_ingestor.utils import (
    load_all_wiki_structures,
    load_page_connections,
    slugify,
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


class ResearchIngestorBase(Ingestor):
    """
    Base class for agentic research ingestors.
    
    Provides:
    - Claude Code agent initialization (api_key auth by default)
    - Wiki structure loading
    - Three-phase pipeline execution (planning, writing, auditing)
    - Page collection from wiki directory
    
    Subclasses only need to implement:
    - source_type property: Return the source type string
    
    The base class handles everything else through the three-phase pipeline.
    
    Example:
        class IdeaIngestor(ResearchIngestorBase):
            @property
            def source_type(self) -> str:
                return "idea"
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the research ingestor.
        
        Args:
            params: Optional parameters:
                - model: Model ID (e.g. "claude-opus-5")
                - effort: Reasoning effort of the phase sessions (pinned, never
                  inherited from the machine's own Claude settings)
                - timeout: Deadline per phase in seconds; None (the default)
                  means a phase runs until it finishes
                - auth_mode: Claude authentication mode (auto, oauth, or api_key)
                - wiki_dir: Output directory (default: data/wikis)
                - staging_subdir: Staging subdirectory (default: "_staging")
                - cleanup_staging: Remove staging after ingest (default: False)
        """
        # Keys not given come from the packaged config's learner.ingestor
        # block, the single source of these defaults.
        super().__init__({**learner_defaults("ingestor"), **(params or {})})
        
        # Agent configuration
        self._timeout = self.params["timeout"]
        self._effort = self.params["effort"]
        self._claude_auth_settings = {"auth_mode": self.params["auth_mode"]}
        self._model = self.params["model"]
        
        # Wiki directory configuration
        self._wiki_dir = Path(self.params.get("wiki_dir", DEFAULT_WIKI_DIR))
        self._staging_subdir = self.params.get("staging_subdir", "_staging")
        self._cleanup_staging = self.params.get("cleanup_staging", False)
        
        # Runtime state
        self._agent = None
        self._staging_dir: Optional[Path] = None
    
    @property
    @abstractmethod
    def source_type(self) -> str:
        """Return the source type this ingestor handles."""
        pass
    
    def _initialize_agent(self, workspace: str) -> None:
        """
        Initialize Claude Code agent with read + write tools.
        
        Args:
            workspace: Path to the workspace directory
        """
        # Base agent_specific config
        agent_specific = {
            "allowed_tools": ["Read", "Write", "Edit", "Bash"],
            "timeout": self._timeout,
            "effort": self._effort,
            "planning_mode": True,
        }
        
        # Model from params (should be provided via config.yaml)
        model = self._model
        
        agent_specific.update(self._claude_auth_settings)
        
        # Build config for Claude Code
        config = CodingAgentFactory.build_config(
            agent_type="claude_code",
            model=model,
            debug_model=model,
            agent_specific=agent_specific,
        )
        
        self._agent = CodingAgentFactory.create(config)
        self._agent.initialize(workspace)
        logger.info(
            f"Initialized Claude Code agent for {workspace} "
            f"(auth={self._claude_auth_settings}, model={model})"
        )
    
    def _normalize_source(self, source: Any) -> Dict[str, Any]:
        """
        Extract query, source URL, and content from input.
        
        Args:
            source: Input source (Idea, Implementation, ResearchReport, or dict)
            
        Returns:
            Dict with query, source_url, and content keys
        """
        if isinstance(source, dict):
            return {
                "query": source.get("query", ""),
                "source_url": source.get("source", ""),
                "content": source.get("content", ""),
            }
        else:
            return {
                "query": getattr(source, "query", ""),
                "source_url": getattr(source, "source", ""),
                "content": getattr(source, "content", ""),
            }
    
    def _ensure_wiki_directories(self) -> None:
        """Ensure wiki subdirectories exist."""
        for subdir in ["principles", "implementations", "environments", "heuristics"]:
            (self._staging_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    def _build_phase_prompt(self, phase: str, **kwargs) -> str:
        """
        Build prompt for a specific phase.
        
        Args:
            phase: Phase name (planning, writing, auditing)
            **kwargs: Variables to substitute in the prompt
            
        Returns:
            Complete prompt string
        """
        # Load base prompt template
        base_prompt = _load_prompt(phase)
        
        # Load wiki structures and connections
        wiki_structures = load_all_wiki_structures()
        page_connections = load_page_connections()
        
        # Add common variables
        kwargs["wiki_structures"] = wiki_structures
        kwargs["page_connections"] = page_connections
        kwargs["wiki_dir"] = str(self._staging_dir)
        kwargs["timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M GMT")
        
        # Format the prompt
        # A placeholder the template names and the caller does not supply is a
        # bug in one of the two; the prompt must never go out unformatted.
        return base_prompt.format(**kwargs)
    
    def _run_phase(self, phase: str, **kwargs) -> None:
        """
        Run a single phase of the pipeline.
        
        Args:
            phase: Phase name
            **kwargs: Variables for the prompt
            
        Raises:
            RuntimeError: If the phase session fails; the later phases build on
                this one's output, so the run stops here.
        """
        start = time.time()
        logger.info(f"Running {phase} phase...")
        
        prompt = self._build_phase_prompt(phase, **kwargs)
        result = self._agent.generate_code(prompt)
        
        elapsed = time.time() - start
        
        if not result.success:
            raise RuntimeError(f"{phase} phase failed after {elapsed:.1f}s: {result.error}")
        
        logger.info(f"{phase} phase complete ({elapsed:.1f}s)")
    
    def _run_planning_phase(self, query: str, source_url: str, content: str) -> None:
        """
        Run Phase 1: Planning.
        
        Analyzes content and writes _plan.md with page decisions.
        """
        return self._run_phase(
            "planning",
            query=query,
            source_url=source_url,
            content=content,
        )
    
    def _run_writing_phase(self, query: str, source_url: str, content: str) -> None:
        """
        Run Phase 2: Writing.
        
        Creates wiki pages based on the plan.
        """
        return self._run_phase(
            "writing",
            query=query,
            source_url=source_url,
            content=content,
        )
    
    def _run_auditing_phase(self) -> None:
        """
        Run Phase 3: Auditing.
        
        Validates pages and fixes issues.
        """
        return self._run_phase("auditing")
    
    def _collect_pages(self) -> List[WikiPage]:
        """
        Collect WikiPage objects from the staging directory.
        
        Returns:
            List of WikiPage objects
        """
        try:
            pages = parse_wiki_directory(self._staging_dir)
            logger.info(f"Collected {len(pages)} pages from {self._staging_dir}")
            return pages
        except Exception as e:
            logger.error(f"Failed to collect pages: {e}")
            return []
    
    def ingest(self, source: Any) -> List[WikiPage]:
        """
        Run the three-phase ingestion pipeline.
        
        1. Planning: Analyze content and decide what pages to create
        2. Writing: Create wiki pages following section definitions
        3. Auditing: Validate pages and fix issues
        
        Args:
            source: Input source (Idea, Implementation, ResearchReport, or dict)
            
        Returns:
            List of WikiPage objects
            
        Raises:
            ValueError: If source has no query
        """
        # Normalize source
        source_data = self._normalize_source(source)
        query = source_data["query"]
        source_url = source_data["source_url"]
        content = source_data["content"]
        
        if not query:
            raise ValueError(f"{self.__class__.__name__} expected a non-empty 'query'")
        
        # Create staging directory
        run_id = uuid.uuid4().hex[:12]
        slug = slugify(query, max_len=30)
        self._staging_dir = self._wiki_dir / self._staging_subdir / f"{self.source_type}_{slug}_{run_id}"
        self._staging_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Staging directory: {self._staging_dir}")
        
        try:
            # Ensure wiki subdirectories exist
            self._ensure_wiki_directories()
            
            # Initialize agent with staging directory as workspace
            self._initialize_agent(str(self._staging_dir))
            
            # Phase 1: Planning
            logger.info("=" * 60)
            logger.info("PHASE 1: Planning")
            logger.info("=" * 60)
            
            self._run_planning_phase(query, source_url, content)
            
            # Phase 2: Writing
            logger.info("=" * 60)
            logger.info("PHASE 2: Writing")
            logger.info("=" * 60)
            
            self._run_writing_phase(query, source_url, content)
            
            # Phase 3: Auditing
            logger.info("=" * 60)
            logger.info("PHASE 3: Auditing")
            logger.info("=" * 60)
            
            self._run_auditing_phase()
            
            # Collect pages
            pages = self._collect_pages()
            
            logger.info(f"Ingestion complete: {len(pages)} pages created")
            return pages
            
        finally:
            # Cleanup staging if requested
            if self._cleanup_staging and self._staging_dir and self._staging_dir.exists():
                import shutil
                shutil.rmtree(self._staging_dir, ignore_errors=True)
                logger.info(f"Cleaned up staging directory: {self._staging_dir}")
    
    def get_staging_dir(self) -> Optional[Path]:
        """
        Get the path to the staging directory.
        
        Useful for debugging or inspecting intermediate results.
        
        Returns:
            Path to staging directory, or None if not set
        """
        return self._staging_dir
