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
    - Claude Code agent initialization (Bedrock by default)
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
                - timeout: Agent timeout in seconds (default: 600)
                - use_bedrock: Use AWS Bedrock (default: True)
                - aws_region: AWS region for Bedrock (default: "us-east-1")
                - model: Model override (default: Sonnet on Bedrock)
                - wiki_dir: Output directory (default: data/wikis)
                - staging_subdir: Staging subdirectory (default: "_staging")
                - cleanup_staging: Remove staging after ingest (default: False)
        """
        super().__init__(params)
        
        # Agent configuration
        self._timeout = self.params.get("timeout", 600)  # 10 minutes default
        self._use_bedrock = self.params.get("use_bedrock", True)  # Bedrock by default
        self._aws_region = self.params.get("aws_region", "us-east-1")
        self._model = self.params.get("model")
        
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
            "planning_mode": True,
        }
        
        # Determine model
        model = self._model
        
        # Configure Bedrock if enabled
        if self._use_bedrock:
            agent_specific["use_bedrock"] = True
            agent_specific["aws_region"] = self._aws_region
            # Default Bedrock model if not specified (Claude Opus 4.5)
            if not model:
                model = "us.anthropic.claude-opus-4-5-20251101-v1:0"
        else:
            # Direct Anthropic API
            if not model:
                model = "claude-sonnet-4-20250514"
        
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
            f"(bedrock={self._use_bedrock}, model={model})"
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
        try:
            return base_prompt.format(**kwargs)
        except KeyError as e:
            logger.warning(f"Missing format variable in {phase} prompt: {e}")
            return base_prompt
    
    def _run_phase(self, phase: str, **kwargs) -> bool:
        """
        Run a single phase of the pipeline.
        
        Args:
            phase: Phase name
            **kwargs: Variables for the prompt
            
        Returns:
            True if phase succeeded, False otherwise
        """
        start = time.time()
        logger.info(f"Running {phase} phase...")
        
        prompt = self._build_phase_prompt(phase, **kwargs)
        result = self._agent.generate_code(prompt)
        
        elapsed = time.time() - start
        
        if not result.success:
            logger.error(f"{phase} phase failed after {elapsed:.1f}s: {result.error}")
            return False
        
        logger.info(f"{phase} phase complete ({elapsed:.1f}s)")
        return True
    
    def _run_planning_phase(self, query: str, source_url: str, content: str) -> bool:
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
    
    def _run_writing_phase(self, query: str, source_url: str, content: str) -> bool:
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
    
    def _run_auditing_phase(self) -> bool:
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
            
            success = self._run_planning_phase(query, source_url, content)
            if not success:
                logger.warning("Planning phase failed, attempting to continue...")
            
            # Phase 2: Writing
            logger.info("=" * 60)
            logger.info("PHASE 2: Writing")
            logger.info("=" * 60)
            
            success = self._run_writing_phase(query, source_url, content)
            if not success:
                logger.warning("Writing phase failed, attempting to collect partial results...")
            
            # Phase 3: Auditing
            logger.info("=" * 60)
            logger.info("PHASE 3: Auditing")
            logger.info("=" * 60)
            
            success = self._run_auditing_phase()
            if not success:
                logger.warning("Auditing phase failed, returning pages without validation...")
            
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
