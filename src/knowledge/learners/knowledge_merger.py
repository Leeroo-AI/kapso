# Knowledge Merger
#
# Merges proposed wiki pages into an existing Knowledge Graph.
# Uses Claude Code agent with wiki MCP tools for intelligent merge decisions.
#
# Architecture:
# - Neo4j: THE Knowledge Graph (nodes + edges)
# - Weaviate: Embedding store for semantic search
# - Source files: Ground truth .md files
#
# Merge Flow:
# 1. Check if KG is indexed in Neo4j
# 2. Group proposed pages by type
# 3. For each page: route to type-specific handler
# 4. Agent searches for related pages, decides merge/create, executes
#
# Usage:
#     from src.knowledge.learners import KnowledgeMerger, MergeInput
#     
#     merger = KnowledgeMerger()
#     result = merger.merge(MergeInput(
#         proposed_pages=pages,
#         main_kg_path=Path("data/wikis"),
#     ))

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.execution.coding_agents.factory import CodingAgentFactory
from src.knowledge.learners.merge_handlers import (
    MergeHandler,
    WorkflowMergeHandler,
    PrincipleMergeHandler,
    ImplementationMergeHandler,
    EnvironmentMergeHandler,
    HeuristicMergeHandler,
)
from src.knowledge.search.base import WikiPage, KGIndexInput
from src.knowledge.search.factory import KnowledgeSearchFactory

# Load merger prompt template
_PROMPT_FILE = Path(__file__).parent / "merger_prompt.md"
MERGE_PROMPT_TEMPLATE = _PROMPT_FILE.read_text() if _PROMPT_FILE.exists() else ""

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class MergeInput:
    """
    Input to the knowledge merger.
    
    Attributes:
        proposed_pages: Pages to merge (from ingestor/staging)
        main_kg_path: Source files directory (ground truth .md files)
        weaviate_collection: Weaviate collection for embeddings
        source_context: Context for logging (e.g., repo URL)
        neo4j_uri: Neo4j connection URI (default: from NEO4J_URI env var)
        neo4j_user: Neo4j username (default: from NEO4J_USER env var)
        neo4j_password: Neo4j password (default: from NEO4J_PASSWORD env var)
    """
    proposed_pages: List[WikiPage]
    main_kg_path: Path
    weaviate_collection: str = "KGWikiPages"
    source_context: str = ""
    neo4j_uri: Optional[str] = None
    neo4j_user: Optional[str] = None
    neo4j_password: Optional[str] = None
    
    def __post_init__(self):
        """Ensure path is Path object and set Neo4j defaults from env."""
        if isinstance(self.main_kg_path, str):
            self.main_kg_path = Path(self.main_kg_path)
        
        # Default to environment variables if not provided
        if self.neo4j_uri is None:
            self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        if self.neo4j_user is None:
            self.neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        if self.neo4j_password is None:
            self.neo4j_password = os.getenv("NEO4J_PASSWORD", "password")


@dataclass
class MergeResult:
    """
    Result of merge operation.
    
    Attributes:
        total_proposed: Number of pages proposed
        created: List of new page IDs created
        merged: List of (proposed_id, target_id) tuples
        errors: List of error messages
    """
    total_proposed: int = 0
    created: List[str] = field(default_factory=list)
    merged: List[Tuple[str, str]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    @property
    def success(self) -> bool:
        """Check if merge completed without critical errors."""
        return len(self.errors) == 0
    
    @property
    def total_processed(self) -> int:
        """Total pages successfully processed."""
        return len(self.created) + len(self.merged)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_proposed": self.total_proposed,
            "created": self.created,
            "merged": self.merged,
            "errors": self.errors,
            "total_processed": self.total_processed,
            "success": self.success,
        }
    
    def __repr__(self) -> str:
        return (
            f"MergeResult(proposed={self.total_proposed}, "
            f"created={len(self.created)}, merged={len(self.merged)}, "
            f"errors={len(self.errors)})"
        )


# =============================================================================
# Knowledge Merger
# =============================================================================

class KnowledgeMerger:
    """
    Merges proposed wiki pages into an existing Knowledge Graph.
    
    The KG is stored in:
    - Neo4j: Graph structure (nodes + edges) - THE INDEX
    - Weaviate: Embeddings for semantic search
    - Source files: Ground truth .md files
    
    Uses Claude Code agent with wiki MCP tools for intelligent merge decisions.
    Each page type has its own MergeHandler with type-specific instructions.
    
    Example:
        from src.knowledge.learners import KnowledgeMerger, MergeInput
        from src.knowledge.search.base import WikiPage
        
        # Prepare pages to merge
        pages = [WikiPage(...), ...]
        
        # Run merge
        merger = KnowledgeMerger()
        result = merger.merge(MergeInput(
            proposed_pages=pages,
            main_kg_path=Path("data/wikis"),
        ))
        
        print(f"Created: {len(result.created)}, Merged: {len(result.merged)}")
    """
    
    def __init__(
        self,
        agent_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize KnowledgeMerger.
        
        Args:
            agent_config: Configuration for Claude Code agent
        """
        self._agent_config = agent_config or {}
        self._agent = None
        self._search_backend = None
        
        # Load handlers from merge_handlers module
        self.handlers = self._load_handlers()
    
    def _load_handlers(self) -> Dict[str, MergeHandler]:
        """Load merge handlers for all page types from merge_handlers module."""
        return {
            "Workflow": WorkflowMergeHandler(),
            "Principle": PrincipleMergeHandler(),
            "Implementation": ImplementationMergeHandler(),
            "Environment": EnvironmentMergeHandler(),
            "Heuristic": HeuristicMergeHandler(),
        }
    
    # =========================================================================
    # Main Merge Entry Point
    # =========================================================================
    
    def merge(self, input: MergeInput) -> MergeResult:
        """
        Main merge entry point.
        
        Process:
        1. Try to initialize search backend (Neo4j + Weaviate)
        2. If no index exists, create all pages as new (no merge needed)
        3. If index exists, use agent to decide merge vs create for each page
        
        Args:
            input: MergeInput with proposed pages and configuration
            
        Returns:
            MergeResult with created, merged, and error counts
        """
        result = MergeResult(total_proposed=len(input.proposed_pages))
        
        if not input.proposed_pages:
            logger.warning("No proposed pages to merge")
            return result
        
        try:
            # Step 1: Check if index is available
            has_index = self._try_initialize_index(input)
            
            if not has_index:
                # No index available - create all pages as new
                logger.info("No existing index. Creating all pages as new...")
                return self._create_all_pages(input, result)
            
            # Step 2: Initialize agent with wiki MCP tools
            self._initialize_agent(input.main_kg_path)
            
            # Step 3: Group pages by type
            by_type = self._group_by_type(input.proposed_pages)
            
            # Step 4: Process each type with merge logic
            for page_type, pages in by_type.items():
                handler = self.handlers.get(page_type)
                if not handler:
                    for p in pages:
                        result.errors.append(f"No handler for type '{page_type}': {p.id}")
                    continue
                
                # Process each page with type-specific handler
                for page in pages:
                    try:
                        action_result = self._process_page(page, handler, input)
                        self._record_action(action_result, result)
                        
                    except Exception as e:
                        error_msg = f"Failed to process {page.id}: {e}"
                        logger.error(error_msg)
                        result.errors.append(error_msg)
            
            logger.info(
                f"Merge complete: {len(result.created)} created, "
                f"{len(result.merged)} merged, {len(result.errors)} errors"
            )
            return result
            
        except Exception as e:
            logger.error(f"Merge failed: {e}")
            result.errors.append(str(e))
            return result
    
    def _try_initialize_index(self, input: MergeInput) -> bool:
        """
        Try to initialize search backend and check if index exists.
        
        Returns:
            True if index is available and has pages, False otherwise
        """
        try:
            self._initialize_search_backend(input)
            return self._is_kg_indexed()
        except Exception as e:
            logger.warning(f"Could not initialize index: {e}")
            return False
    
    def _create_all_pages(self, input: MergeInput, result: MergeResult) -> MergeResult:
        """
        Create all proposed pages as new (no merge).
        
        Used when there's no existing index to merge with.
        1. Writes pages to wiki directory organized by type
        2. Indexes pages via search backend (Neo4j + Weaviate if available)
        """
        wiki_dir = input.main_kg_path
        wiki_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Write pages to wiki directory
        for page in input.proposed_pages:
            try:
                self._write_page_to_wiki(page, wiki_dir)
                result.created.append(page.id)
                logger.info(f"Created new page: {page.id}")
                
            except Exception as e:
                error_msg = f"Failed to create {page.id}: {e}"
                logger.error(error_msg)
                result.errors.append(error_msg)
        
        logger.info(f"Created {len(result.created)} new pages")
        
        # Step 2: Index pages via search backend
        try:
            # Initialize search backend if not already done
            if not self._search_backend:
                self._initialize_search_backend(input)
            
            # Index the pages (will gracefully skip if Neo4j/Weaviate unavailable)
            index_input = KGIndexInput(
                pages=input.proposed_pages,
                wiki_dir=wiki_dir,
            )
            self._search_backend.index(index_input)
            logger.info(f"Indexed {len(input.proposed_pages)} pages to search backend")
            
        except Exception as e:
            logger.warning(f"Could not index pages to search backend: {e}")
            # Not a critical error - pages are still written to disk
        
        return result
    
    def _write_page_to_wiki(self, page: "WikiPage", wiki_dir: Path) -> None:
        """
        Write a WikiPage to the wiki directory.
        
        Organizes pages into type subdirectories:
        - workflows/
        - principles/
        - implementations/
        - environments/
        - heuristics/
        """
        # Map page type to subdirectory
        type_to_subdir = {
            "Workflow": "workflows",
            "Principle": "principles",
            "Implementation": "implementations",
            "Environment": "environments",
            "Heuristic": "heuristics",
        }
        
        subdir = type_to_subdir.get(page.page_type, "other")
        type_dir = wiki_dir / subdir
        type_dir.mkdir(parents=True, exist_ok=True)
        
        # Write page content to file
        filename = f"{page.page_title}.md"
        file_path = type_dir / filename
        file_path.write_text(page.content, encoding="utf-8")
        
        logger.debug(f"Wrote {file_path}")
    
    # =========================================================================
    # Search Backend Management
    # =========================================================================
    
    def _initialize_search_backend(self, input: MergeInput) -> None:
        """Initialize the search backend for KG operations."""
        self._search_backend = KnowledgeSearchFactory.create(
            "kg_graph_search",
            params={
                "weaviate_collection": input.weaviate_collection,
                "neo4j_uri": input.neo4j_uri,
                "neo4j_user": input.neo4j_user,
                "neo4j_password": input.neo4j_password,
            },
        )
        logger.info(f"Initialized search backend with collection: {input.weaviate_collection}")
    
    def _is_kg_indexed(self) -> bool:
        """
        Check if KG exists in Neo4j (has WikiPage nodes).
        
        Returns:
            True if KG has indexed pages, False otherwise
        """
        if not self._search_backend:
            return False
        
        # Try to access the neo4j driver from the search backend
        try:
            driver = getattr(self._search_backend, "_neo4j_driver", None)
            if not driver:
                logger.warning("Neo4j driver not available")
                return False
            
            with driver.session() as session:
                result = session.run("MATCH (p:WikiPage) RETURN count(p) as count")
                count = result.single()["count"]
                logger.info(f"KG has {count} WikiPage nodes in Neo4j")
                return count > 0
                
        except Exception as e:
            logger.warning(f"Could not check KG index status: {e}")
            return False
    
    def _run_initial_index(self, main_kg_path: Path) -> None:
        """
        Index main KG from source files to Neo4j + Weaviate.
        
        Args:
            main_kg_path: Path to wiki source files
        """
        if not self._search_backend:
            raise RuntimeError("Search backend not initialized")
        
        if not main_kg_path.exists():
            logger.warning(f"Main KG path does not exist: {main_kg_path}")
            return
        
        logger.info(f"Indexing main KG from: {main_kg_path}")
        
        self._search_backend.index(KGIndexInput(wiki_dir=main_kg_path))
        
        logger.info("Initial index complete")
    
    # =========================================================================
    # Agent Management
    # =========================================================================
    
    def _initialize_agent(self, workspace: Path) -> None:
        """
        Initialize Claude Code agent with wiki MCP tools.
        
        Args:
            workspace: Working directory for the agent
        """
        try:
            # Build config for Claude Code agent with wiki tools
            config = CodingAgentFactory.build_config(
                agent_type="claude_code",
                agent_specific={
                    # Wiki MCP tools for KG operations
                    "allowed_tools": [
                        "Read",
                        "mcp__kg-graph-search__search_knowledge",
                        "mcp__kg-graph-search__get_wiki_page",
                        "mcp__kg-graph-search__kg_index",
                        "mcp__kg-graph-search__kg_edit",
                    ],
                    "timeout": self._agent_config.get("timeout", 1800),
                }
            )
            
            self._agent = CodingAgentFactory.create(config)
            self._agent.initialize(str(workspace))
            logger.info("Initialized Claude Code agent with wiki MCP tools")
            
        except ImportError as e:
            logger.error(f"Could not import CodingAgentFactory: {e}")
            raise RuntimeError(
                "Claude Code agent not available. "
                "Ensure coding_agents module is properly installed."
            )
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            raise
    
    # =========================================================================
    # Page Processing
    # =========================================================================
    
    def _group_by_type(self, pages: List[WikiPage]) -> Dict[str, List[WikiPage]]:
        """Group pages by their page_type."""
        by_type: Dict[str, List[WikiPage]] = {}
        for page in pages:
            page_type = page.page_type
            if page_type not in by_type:
                by_type[page_type] = []
            by_type[page_type].append(page)
        return by_type
    
    def _process_page(
        self,
        page: WikiPage,
        handler: MergeHandler,
        input: MergeInput,
    ) -> Dict[str, Any]:
        """
        Process a single page using Claude Code agent.
        
        Steps:
        1. Build prompt with handler's merge instructions
        2. Agent uses search_knowledge to find related pages
        3. Agent uses get_wiki_page to read candidates
        4. Agent decides: create new OR merge with existing
        5. Agent executes: kg_index OR kg_edit
        
        Args:
            page: Proposed page to merge
            handler: Type-specific merge handler
            input: Merge input configuration
            
        Returns:
            Dict with action ("created" or "merged") and details
        """
        if not self._agent:
            raise RuntimeError("Agent not initialized")
        
        # Build prompt
        prompt = self._build_agent_prompt(page, handler, input)
        
        logger.info(f"Processing {page.id} with {handler.page_type} handler...")
        
        # Run agent
        agent_result = self._agent.generate_code(prompt)
        
        if not agent_result.success:
            raise RuntimeError(f"Agent failed: {agent_result.error}")
        
        # Parse agent result to determine action taken
        return self._parse_agent_result(agent_result.output, page.id)
    
    def _build_agent_prompt(
        self,
        page: WikiPage,
        handler: MergeHandler,
        input: MergeInput,
    ) -> str:
        """
        Build prompt for Claude Code agent.
        
        Args:
            page: Proposed page to merge
            handler: Type-specific merge handler
            input: Merge input configuration
            
        Returns:
            Complete prompt string
        """
        # Format page content (truncate if too long)
        content_preview = page.content[:4000] if len(page.content) > 4000 else page.content
        if len(page.content) > 4000:
            content_preview += "\n\n... [content truncated]"
        
        # Format outgoing links
        links_json = json.dumps(page.outgoing_links, indent=2) if page.outgoing_links else "[]"
        
        # Build search query for the agent to use
        search_query = handler.build_search_query(page)
        search_filters = handler.get_search_filters()
        
        # Format prompt from template
        prompt = MERGE_PROMPT_TEMPLATE.format(
            main_kg_path=input.main_kg_path,
            weaviate_collection=input.weaviate_collection,
            page_id=page.id,
            page_type=page.page_type,
            page_title=page.page_title,
            overview=page.overview,
            domains=", ".join(page.domains) if page.domains else "None",
            outgoing_links=links_json,
            content=content_preview,
            merge_instructions=handler.merge_instructions,
            search_query=search_query,
            search_page_types=json.dumps(search_filters.get("page_types", [])),
            search_top_k=search_filters.get("top_k", 5),
        )
        
        return prompt
    
    def _parse_agent_result(
        self,
        output: str,
        proposed_id: str,
    ) -> Dict[str, Any]:
        """
        Parse agent output to determine action taken.
        
        Args:
            output: Agent output text
            proposed_id: ID of proposed page
            
        Returns:
            Dict with action and details
        """
        output_upper = output.upper()
        
        # Look for ACTION: CREATED or ACTION: MERGED
        if "ACTION: CREATED" in output_upper:
            return {"action": "created", "page_id": proposed_id}
        
        elif "ACTION: MERGED" in output_upper:
            # Try to find TARGET: <page_id>
            target_id = None
            for line in output.split("\n"):
                if line.upper().startswith("TARGET:"):
                    target_id = line.split(":", 1)[1].strip()
                    break
            
            return {
                "action": "merged",
                "proposed_id": proposed_id,
                "target_id": target_id or "unknown",
            }
        
        # Default: assume created if we can't parse
        logger.warning(f"Could not parse agent action for {proposed_id}, assuming created")
        return {"action": "created", "page_id": proposed_id}
    
    def _record_action(
        self,
        action_result: Dict[str, Any],
        result: MergeResult,
    ) -> None:
        """
        Record action result in MergeResult.
        
        Args:
            action_result: Dict from _parse_agent_result
            result: MergeResult to update
        """
        action = action_result.get("action")
        
        if action == "created":
            page_id = action_result.get("page_id", "unknown")
            result.created.append(page_id)
            logger.info(f"Created new page: {page_id}")
            
        elif action == "merged":
            proposed_id = action_result.get("proposed_id", "unknown")
            target_id = action_result.get("target_id", "unknown")
            result.merged.append((proposed_id, target_id))
            logger.info(f"Merged {proposed_id} into {target_id}")
    
    # =========================================================================
    # Cleanup
    # =========================================================================
    
    def close(self) -> None:
        """Clean up resources."""
        if self._search_backend:
            self._search_backend.close()
            self._search_backend = None
        
        self._agent = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False


# =============================================================================
# CLI Test
# =============================================================================

if __name__ == "__main__":
    """Test the KnowledgeMerger with sample data."""
    print("=" * 60)
    print("KnowledgeMerger Test")
    print("=" * 60)
    
    # Create test proposed pages
    test_pages = [
        WikiPage(
            id="Principle/Test_Principle",
            page_title="Test_Principle",
            page_type="Principle",
            overview="A test principle for the merger",
            content="== Overview ==\nTest content for principle.",
            domains=["Test"],
            outgoing_links=[],
        ),
    ]
    
    # Test merger
    merger = KnowledgeMerger()
    
    print(f"\nTest with {len(test_pages)} proposed pages")
    print("-" * 60)
    
    input_data = MergeInput(
        proposed_pages=test_pages,
        main_kg_path=Path("data/wikis"),
    )
    
    result = merger.merge(input_data)
    
    print(f"\nResult: {result}")
    print(f"  Created: {len(result.created)}")
    print(f"  Merged: {len(result.merged)}")
    print(f"  Errors: {len(result.errors)}")
    
    print("\n" + "=" * 60)
    print("Test complete!")
