# Knowledge Merger
#
# Hierarchical sub-graph-aware knowledge merger.
# Uses a single Claude Code agent call with comprehensive instructions
# to merge proposed pages into the Knowledge Graph.
#
# Architecture:
# - Neo4j: THE Knowledge Graph (nodes + edges)
# - Weaviate: Embedding store for semantic search
# - Source files: Ground truth .md files
#
# Merge Flow:
# 1. Check if KG is indexed
# 2. If no index: create all pages as new
# 3. If index exists: run agentic hierarchical merge
#    - Phase 1: Detect sub-graphs
#    - Phase 2: Plan (top-down)
#    - Phase 3: Execute (bottom-up)
#    - Phase 4: Audit
#    - Phase 5: Finalize
#
# Usage:
#     from kapso.knowledge_base.learners.merger import KnowledgeMerger
#     
#     merger = KnowledgeMerger()
#     result = merger.merge(pages, wiki_dir=Path("data/wikis"))

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from kapso.execution.coding_agents.factory import CodingAgentFactory
from kapso.knowledge_base.learners.merger.prompts import load_prompt
from kapso.knowledge_base.search.base import WikiPage, KGIndexMetadata

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class MergeResult:
    """
    Result of hierarchical merge operation.
    
    Attributes:
        total_proposed: Number of pages proposed for merge
        subgraphs_processed: Number of sub-graphs detected and processed
        created: List of new page IDs created
        edited: List of page IDs that were merged/edited
        failed: List of page IDs that failed to process
        errors: List of error messages
        plan_path: Path to the merge plan file
    """
    total_proposed: int = 0
    subgraphs_processed: int = 0
    created: List[str] = field(default_factory=list)
    edited: List[str] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    plan_path: Optional[Path] = None
    
    @property
    def success(self) -> bool:
        """Check if merge completed without critical errors."""
        return len(self.errors) == 0 and len(self.failed) == 0
    
    @property
    def total_processed(self) -> int:
        """Total pages successfully processed."""
        return len(self.created) + len(self.edited)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_proposed": self.total_proposed,
            "subgraphs_processed": self.subgraphs_processed,
            "created": self.created,
            "edited": self.edited,
            "failed": self.failed,
            "errors": self.errors,
            "total_processed": self.total_processed,
            "success": self.success,
            "plan_path": str(self.plan_path) if self.plan_path else None,
        }
    
    def __repr__(self) -> str:
        return (
            f"MergeResult(proposed={self.total_proposed}, "
            f"subgraphs={self.subgraphs_processed}, "
            f"created={len(self.created)}, edited={len(self.edited)}, "
            f"failed={len(self.failed)}, errors={len(self.errors)})"
        )


# =============================================================================
# Knowledge Merger
# =============================================================================

class KnowledgeMerger:
    """
    Hierarchical sub-graph-aware knowledge merger.
    
    Uses a single Claude Code agent call with comprehensive instructions
    to merge proposed pages into the Knowledge Graph.
    
    The KG is stored in:
    - Neo4j: Graph structure (nodes + edges) - THE INDEX
    - Weaviate: Embeddings for semantic search
    - Source files: Ground truth .md files
    
    Default configuration uses AWS Bedrock with Claude Opus 4.5.
    
    Example:
        from kapso.knowledge_base.learners.merger import KnowledgeMerger
        from kapso.knowledge_base.search.base import WikiPage
        
        # Prepare pages to merge
        pages = [WikiPage(...), ...]
        
        # Run merge (uses Bedrock by default)
        merger = KnowledgeMerger()
        result = merger.merge(pages, wiki_dir=Path("data/wikis"))
        
        print(f"Created: {len(result.created)}, Edited: {len(result.edited)}")
        print(f"Plan: {result.plan_path}")
    """
    
    # Maximum retry attempts for failed sub-graphs
    MAX_RETRIES = 3
    
    def __init__(self, agent_config: Optional[Dict[str, Any]] = None):
        """
        Initialize KnowledgeMerger.
        
        Args:
            agent_config: Configuration for Claude Code agent. Supports:
                - kg_index_path: Path to .index file for KG backend config
                - timeout: Agent timeout in seconds (default: 3600)
                - use_bedrock: Use AWS Bedrock (default: True)
                - aws_region: AWS region for Bedrock
                - model: Model ID override
        """
        self._agent_config = agent_config or {}
        self._kg_index_path: Optional[str] = self._agent_config.get("kg_index_path")
        self._agent = None
    
    # =========================================================================
    # Main Merge Entry Point
    # =========================================================================
    
    def merge(self, proposed_pages: List[WikiPage], wiki_dir: Path) -> MergeResult:
        """
        Main merge entry point using hierarchical sub-graph-aware algorithm.
        
        Process:
        1. Check if KG index exists (explicit path or auto-detect in wiki_dir)
        2. If no index: create all pages as new (write to wiki_dir)
        3. If index exists: run agentic hierarchical merge
        
        Args:
            proposed_pages: Proposed pages to add/merge into the KG
            wiki_dir: Persistent wiki directory on disk (KG source-of-truth)
            
        Returns:
            MergeResult with created, edited, failed, and error counts
        """
        wiki_dir = (Path(wiki_dir) if isinstance(wiki_dir, str) else wiki_dir).expanduser().resolve()
        result = MergeResult(total_proposed=len(proposed_pages))
        
        if not proposed_pages:
            logger.warning("No proposed pages to merge")
            return result
        
        try:
            # Step 1: Check if index is available (explicit or auto-detect)
            has_index = self._try_initialize_index(wiki_dir)
            
            if not has_index:
                # No index available - create all pages as new
                logger.info("No existing index. Creating all pages as new...")
                return self._create_all_pages(proposed_pages, wiki_dir, result)
            
            # Step 2: Run agentic hierarchical merge
            return self._run_agentic_merge(proposed_pages, wiki_dir)
            
        except Exception as e:
            logger.error(f"Merge failed: {e}")
            result.errors.append(str(e))
            return result
    
    def _try_initialize_index(self, wiki_dir: Path) -> bool:
        """
        Check if we should use merge mode (agent + MCP tools).
        
        Checks in order:
        1. Explicit kg_index_path from agent_config
        2. Auto-detect .index file in wiki_dir
        
        Args:
            wiki_dir: Wiki directory to check for .index file
        
        Returns:
            True if index exists and merge mode should be used
        """
        # Priority 1: Explicit path from config
        index_path_to_check = self._kg_index_path
        
        # Priority 2: Auto-detect in wiki_dir
        if not index_path_to_check:
            auto_index = wiki_dir / ".index"
            if auto_index.exists():
                logger.info(f"Auto-detected index file: {auto_index}")
                index_path_to_check = str(auto_index)
                self._kg_index_path = index_path_to_check
        
        if not index_path_to_check:
            return False
        
        try:
            index_path = Path(index_path_to_check).expanduser().resolve()
            if not index_path.exists():
                raise FileNotFoundError(f"Index file not found: {index_path}")
            
            index_data = json.loads(index_path.read_text(encoding="utf-8"))
            metadata = KGIndexMetadata.from_dict(index_data)
            
            backend = (metadata.search_backend or "").strip()
            if backend.lower() != "kg_graph_search":
                raise NotImplementedError(
                    f"KnowledgeMerger only supports 'kg_graph_search' backend. "
                    f"Got: {backend!r}"
                )
            
            logger.info(f"Using index: {index_path} ({metadata.page_count} pages)")
            return True
            
        except Exception as e:
            raise RuntimeError(f"Invalid kg_index_path={index_path_to_check!r}: {e}") from e
    
    # =========================================================================
    # Create All Pages (No Index Mode)
    # =========================================================================
    
    def _create_all_pages(
        self,
        proposed_pages: List[WikiPage],
        wiki_dir: Path,
        result: MergeResult,
    ) -> MergeResult:
        """
        Create all proposed pages as new (no merge).
        
        Used when there's no existing index to merge with.
        """
        wiki_dir.mkdir(parents=True, exist_ok=True)
        
        # Write pages to wiki directory
        for page in proposed_pages:
            try:
                self._write_page_to_wiki(page, wiki_dir)
                result.created.append(page.id)
                logger.info(f"Created new page: {page.id}")
                
            except Exception as e:
                error_msg = f"Failed to create {page.id}: {e}"
                logger.error(error_msg)
                result.errors.append(error_msg)
                result.failed.append(page.id)
        
        logger.info(f"Created {len(result.created)} new pages")
        
        # Index pages using Kapso.index_kg() - creates .index file in wiki_dir
        # This enables auto-detection for subsequent merge calls
        try:
            from kapso.kapso import Kapso
            
            index_path = wiki_dir / ".index"
            kapso = Kapso()
            kapso.index_kg(
                wiki_dir=str(wiki_dir),
                save_to=str(index_path),
            )
            logger.info(f"Created index file: {index_path}")
            
        except Exception as e:
            logger.warning(f"Could not create index file: {e}")
        
        return result
    
    def _write_page_to_wiki(self, page: WikiPage, wiki_dir: Path) -> None:
        """Write a WikiPage to the wiki directory."""
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
        
        filename = f"{page.page_title}.md"
        file_path = type_dir / filename
        file_path.write_text(page.content, encoding="utf-8")
        
        logger.debug(f"Wrote {file_path}")
    
    # =========================================================================
    # Agentic Hierarchical Merge
    # =========================================================================
    
    def _run_agentic_merge(
        self,
        pages: List[WikiPage],
        wiki_dir: Path,
    ) -> MergeResult:
        """
        Execute the single-agent hierarchical merge.
        
        The agent receives all pages and comprehensive instructions,
        then executes the 5-phase merge algorithm autonomously.
        """
        # Initialize agent
        self._initialize_agent(wiki_dir)
        
        # Build comprehensive prompt
        prompt = self._build_merge_prompt(pages, wiki_dir)
        
        logger.info(f"Running hierarchical merge for {len(pages)} pages...")
        
        # Single agent call
        try:
            agent_result = self._agent.generate_code(prompt)
        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            import traceback
            traceback.print_exc()
            result = MergeResult(total_proposed=len(pages))
            result.errors.append(f"Agent execution failed: {e}")
            return result
        
        if not agent_result.success:
            result = MergeResult(total_proposed=len(pages))
            result.errors.append(f"Agent failed: {agent_result.error}")
            return result
        
        # Parse results from plan.md
        result = self._parse_merge_plan(wiki_dir)
        result.total_proposed = len(pages)
        
        logger.info(
            f"Merge complete: {len(result.created)} created, "
            f"{len(result.edited)} edited, {len(result.errors)} errors"
        )
        
        return result
    
    def _initialize_agent(self, workspace: Path) -> None:
        """
        Initialize Claude Code agent with wiki MCP tools.
        
        Defaults to AWS Bedrock with Claude Opus 4.5 if no config provided.
        
        MCP Server Configuration:
        - Configures the kg-graph-search MCP server for knowledge operations
        - Passes KG_INDEX_PATH to the MCP server via environment
        """
        # Get project root for MCP server paths
        # The MCP server module is at src.gated_mcp.server
        project_root = Path(__file__).parent.parent.parent.parent.parent
        
        # Build MCP server configuration
        mcp_env = {
            "PYTHONPATH": str(project_root),
            "MCP_ENABLED_GATES": "kg",  # Only enable KG gate for merger
        }
        
        # Pass KG index path to MCP server if available
        if self._kg_index_path:
            mcp_env["KG_INDEX_PATH"] = str(self._kg_index_path)
        
        mcp_servers = {
            "kg-graph-search": {
                "command": "python",
                "args": ["-m", "kapso.gated_mcp.server"],
                "cwd": str(project_root),
                "env": mcp_env,
            }
        }
        
        agent_specific = {
            "allowed_tools": [
                "Read",
                "Write",
                "mcp__kg-graph-search__search_knowledge",
                "mcp__kg-graph-search__get_wiki_page",
                "mcp__kg-graph-search__get_page_structure",
                "mcp__kg-graph-search__kg_index",
                "mcp__kg-graph-search__kg_edit",
            ],
            "timeout": self._agent_config.get("timeout", 3600),
            "planning_mode": True,
            "mcp_servers": mcp_servers,
        }
        
        # Model configuration
        model = self._agent_config.get("model")
        
        # Default to Bedrock if not explicitly disabled
        use_bedrock = self._agent_config.get("use_bedrock", True)
        
        if use_bedrock:
            agent_specific["use_bedrock"] = True
            if self._agent_config.get("aws_region"):
                agent_specific["aws_region"] = self._agent_config["aws_region"]
            if not model:
                model = "us.anthropic.claude-opus-4-5-20251101-v1:0"
        
        config = CodingAgentFactory.build_config(
            agent_type="claude_code",
            model=model,
            debug_model=model,
            agent_specific=agent_specific,
        )
        
        self._agent = CodingAgentFactory.create(config)
        self._agent.initialize(str(workspace))
        logger.info(f"Initialized Claude Code agent (bedrock={use_bedrock}, mcp=True)")
    
    def _build_merge_prompt(self, pages: List[WikiPage], wiki_dir: Path) -> str:
        """Build the comprehensive merge instruction prompt."""
        # Load prompt template
        template = load_prompt("hierarchical_merge")
        
        # Serialize pages
        serialized = self._serialize_pages(pages)
        
        # Format prompt
        prompt = template.format(
            wiki_dir=str(wiki_dir),
            max_retries=self.MAX_RETRIES,
            serialized_pages=serialized,
            timestamp=datetime.now().isoformat(),
        )
        
        return prompt
    
    def _serialize_pages(self, pages: List[WikiPage]) -> str:
        """
        Serialize pages to markdown format for prompt context.
        """
        parts = []
        
        for page in pages:
            parts.append(f"### Page: {page.id}")
            parts.append(f"- **Type**: {page.page_type}")
            parts.append(f"- **Title**: {page.page_title}")
            parts.append(f"- **Overview**: {page.overview}")
            parts.append(f"- **Domains**: {', '.join(page.domains) if page.domains else 'None'}")
            
            if page.outgoing_links:
                parts.append("- **Outgoing Links**:")
                for link in page.outgoing_links:
                    edge = link.get('edge_type', 'related')
                    target_type = link.get('target_type', '')
                    target_id = link.get('target_id', '')
                    parts.append(f"  - `{edge}` → {target_type}:{target_id}")
            else:
                parts.append("- **Outgoing Links**: None")
            
            # Content (truncated if too long)
            parts.append("")
            parts.append("**Content:**")
            parts.append("```")
            content = page.content
            if len(content) > 3000:
                content = content[:3000] + "\n... [content truncated]"
            parts.append(content)
            parts.append("```")
            parts.append("")
            parts.append("---")
            parts.append("")
        
        return "\n".join(parts)
    
    def _parse_merge_plan(self, wiki_dir: Path) -> MergeResult:
        """
        Parse the plan.md file to extract merge results.
        """
        plan_path = wiki_dir / "_merge_plan.md"
        result = MergeResult(plan_path=plan_path)
        
        if not plan_path.exists():
            result.errors.append("Plan file not found: _merge_plan.md")
            return result
        
        content = plan_path.read_text(encoding="utf-8")
        
        # Parse created pages
        created_match = re.search(
            r'### Created Pages\s*\n(.*?)(?=###|\Z)',
            content,
            re.DOTALL
        )
        if created_match:
            for line in created_match.group(1).strip().split('\n'):
                line = line.strip()
                if line.startswith('- '):
                    page_id = line[2:].strip()
                    if page_id and not page_id.startswith('('):
                        result.created.append(page_id)
        
        # Parse edited pages
        edited_match = re.search(
            r'### Edited Pages\s*\n(.*?)(?=###|\Z)',
            content,
            re.DOTALL
        )
        if edited_match:
            for line in edited_match.group(1).strip().split('\n'):
                line = line.strip()
                if line.startswith('- '):
                    page_id = line[2:].strip()
                    if page_id and not page_id.startswith('('):
                        result.edited.append(page_id)
        
        # Parse failed pages
        failed_match = re.search(
            r'### Failed Pages\s*\n(.*?)(?=###|\Z)',
            content,
            re.DOTALL
        )
        if failed_match:
            for line in failed_match.group(1).strip().split('\n'):
                line = line.strip()
                if line.startswith('- '):
                    # Format: "- page_id - reason" or just "- page_id"
                    parts = line[2:].split(' - ', 1)
                    page_id = parts[0].strip()
                    if page_id and not page_id.startswith('('):
                        result.failed.append(page_id)
                        if len(parts) > 1:
                            result.errors.append(f"{page_id}: {parts[1]}")
        
        # Parse status
        status_match = re.search(r'### Status:\s*(\w+)', content)
        if status_match:
            status = status_match.group(1).upper()
            if status == "FAILED":
                result.errors.append("Merge failed - see plan.md for details")
        
        # Count subgraphs
        subgraph_matches = re.findall(r'### SubGraph \d+', content)
        result.subgraphs_processed = len(subgraph_matches)
        
        return result
    
    # =========================================================================
    # Cleanup
    # =========================================================================
    
    def close(self) -> None:
        """Clean up resources."""
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
            outgoing_links=[
                {
                    "edge_type": "implemented_by",
                    "target_type": "Implementation",
                    "target_id": "Test_Implementation",
                }
            ],
        ),
        WikiPage(
            id="Implementation/Test_Implementation",
            page_title="Test_Implementation",
            page_type="Implementation",
            overview="A test implementation",
            content="== Overview ==\nTest implementation content.",
            domains=["Test"],
            outgoing_links=[],
        ),
    ]
    
    # Test merger (no index - will create all as new)
    merger = KnowledgeMerger()
    
    print(f"\nTest with {len(test_pages)} proposed pages")
    print("-" * 60)
    
    result = merger.merge(test_pages, wiki_dir=Path("data/wikis_test"))
    
    print(f"\nResult: {result}")
    print(f"  Created: {result.created}")
    print(f"  Edited: {result.edited}")
    print(f"  Failed: {result.failed}")
    print(f"  Errors: {result.errors}")
    
    print("\n" + "=" * 60)
    print("Test complete!")
