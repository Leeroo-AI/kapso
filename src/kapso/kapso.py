# Kapso Agent - Main Entry Point
#
# The primary user-facing API for the Kapso Agent system.
# Provides a clean interface for the "Brain to Binary" workflow:
#   Kapso.index_kg() -> Kapso.evolve() -> Kapso.deploy() -> Software.run()
#
# Usage:
#     from kapso.kapso import Kapso, Source, DeployStrategy
#     
#     # One-time setup: Index knowledge graph
#     kapso = Kapso(config_path="./config.yaml")
#     kapso.index_kg(wiki_dir="data/wikis/ml_knowledge", save_to="data/indexes/ml.index")
#     
#     # Normal usage: Load existing index
#     kapso = Kapso(config_path="./config.yaml", kg_index="data/indexes/ml.index")
#     solution = kapso.evolve(goal="Create a triage agent")
#     software = kapso.deploy(solution, strategy=DeployStrategy.LOCAL)
#     result = software.run({"input": "data"})

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Load environment variables from .env file (if present)
from dotenv import load_dotenv
load_dotenv()

from kapso.execution.orchestrator import OrchestratorAgent
from kapso.execution.solution import SolutionResult
from kapso.environment.handlers.generic import GenericProblemHandler
from kapso.knowledge_base.search import KnowledgeSearchFactory, KGIndexInput
from kapso.knowledge_base.search.base import KGIndexMetadata
from kapso.knowledge_base.learners import Source, KnowledgePipeline
from kapso.researcher import Researcher, ResearchDepth, ResearchMode
from kapso.knowledge_base.types import ResearchFindings
from kapso.core.config import load_config

# Placeholder types for unimplemented learning
class KnowledgeChunk:
    pass

LearnerFactory = None  # Learning not implemented yet
from kapso.deployment import (
    Software,
    DeployConfig,
    DeployStrategy,
    DeploymentFactory,
)


# =============================================================================
# EXCEPTIONS
# =============================================================================

class KGIndexError(Exception):
    """
    Raised when KG index file is invalid or backend data is missing.
    
    This typically happens when:
    - The .index file exists but the backend (Weaviate/Neo4j) was wiped
    - The .index file is corrupted or has invalid format
    - The backend is not accessible
    """
    pass


# =============================================================================
# KAPSO AGENT
# =============================================================================

# Path to default configuration
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


class Kapso:
    """
    The main Kapso Agent class.
    
    A Kapso is an intelligent agent that can:
    1. Index knowledge from wiki pages or JSON knowledge graphs
    2. Evolve software to solve goals using experimentation
    3. Deploy solutions as running software
    
    Knowledge Graph Workflow:
        # ONE-TIME SETUP: Index your knowledge
        kapso = Kapso(config_path="./config.yaml")
        kapso.index_kg(
            wiki_dir="data/wikis/ml_knowledge",
            save_to="data/indexes/ml.index",
        )
        
        # EVERY TIME: Load existing index
        kapso = Kapso(
            config_path="./config.yaml",
            kg_index="data/indexes/ml.index",
        )
        solution = kapso.evolve(goal="Create a momentum trading bot")
        software = kapso.deploy(solution)
        result = software.run({"ticker": "AAPL"})
        
    Advanced usage with evaluation and data directories:
        solution = kapso.evolve(
            goal="Build a classifier with 95% accuracy",
            eval_dir="./evaluation/",
            data_dir="./datasets/",
            initial_repo="https://github.com/owner/starter-repo",
        )
    """
    
    # Mapping from Source type to Learner type
    _SOURCE_TO_LEARNER = {
        Source.Repo: "repo",
        Source.Solution: "experiment",
    }
    
    def __init__(
        self, 
        config_path: Optional[str] = None,
        kg_index: Optional[str] = None,
    ):
        """
        Initialize a Kapso agent.
        
        Args:
            config_path: Path to configuration file (uses default if not provided)
            kg_index: Path to existing .index file to load knowledge graph from.
                      If provided, connects to the indexed knowledge graph.
                      If not provided, knowledge search is disabled.
        """
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self._config = load_config(self.config_path)
        
        # Track learned knowledge chunks (in-memory for MVP)
        self._learned_chunks: List[KnowledgeChunk] = []
        
        # Initialize knowledge search
        if kg_index:
            self._load_kg_index(kg_index)
            self._kg_index_path = kg_index
        else:
            self.knowledge_search = KnowledgeSearchFactory.create_null()
            self._kg_index_path = None
        
        # Print initialization status
        if kg_index:
            print(f"Initialized Kapso")
        else:
            print(f"Initialized Kapso (Knowledge Graph: disabled)")

        # Lazy-initialized web researcher (created on first `.research()` call).
        self._web_researcher: Optional[Researcher] = None
    
    # =========================================================================
    # Knowledge Graph Indexing
    # =========================================================================
    
    def _load_kg_index(self, index_path: str) -> None:
        """
        Load existing index from .index file.
        
        Args:
            index_path: Path to the .index file
            
        Raises:
            KGIndexError: If index file is invalid or backend data is missing
            FileNotFoundError: If index file doesn't exist
        """
        index_path = Path(index_path)
        
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        
        # Load index metadata
        with open(index_path) as f:
            index_data = json.load(f)
        
        metadata = KGIndexMetadata.from_dict(index_data)
        
        # Get search config from mode config
        mode = self._config.get("default_mode", "GENERIC")
        mode_config = self._config.get("modes", {}).get(mode, {})
        search_config = mode_config.get("knowledge_search", {})
        
        # Merge backend_refs into params (backend_refs take precedence)
        params = search_config.get("params", {}).copy()
        params.update(metadata.backend_refs)
        
        # Create search backend
        self.knowledge_search = KnowledgeSearchFactory.create(
            search_type=metadata.search_backend,
            params=params,
        )
        
        # Validate backend has data
        if not self.knowledge_search.validate_backend_data():
            raise KGIndexError(
                f"Index file exists but backend data not found.\n"
                f"Re-index with: kapso.index_kg("
                f"wiki_dir='{metadata.data_source}', save_to='{index_path}')"
            )
        
        print(f"  Knowledge Graph: Loaded ({metadata.page_count} pages from {metadata.search_backend})")
    
    def index_kg(
        self,
        wiki_dir: Optional[str] = None,
        data_path: Optional[str] = None,
        save_to: str = None,
        search_type: Optional[str] = None,
        force: bool = False,
    ) -> str:
        """
        Index knowledge data and save index reference file.
        
        This is a ONE-TIME operation. After indexing, the data persists
        in the configured backends (Weaviate, Neo4j, etc.). Use the returned
        .index file path with kg_index parameter on subsequent runs.
        
        Args:
            wiki_dir: Path to wiki directory (for kg_graph_search backend).
                      Contains .md files organized in type subdirectories.
            data_path: Path to JSON data file (for kg_llm_navigation backend).
                       Contains nodes and edges dict.
            save_to: Path to save .index file (e.g., "data/indexes/ml.index")
            search_type: Override search backend type. If not provided, uses
                         config default or infers from input type.
            force: If True, clears existing data before indexing
            
        Returns:
            Path to created .index file
            
        Raises:
            ValueError: If neither wiki_dir nor data_path provided
            
        Example:
            # Index wiki pages (kg_graph_search)
            kapso.index_kg(
                wiki_dir="data/wikis/ml_knowledge",
                save_to="data/indexes/ml.index",
            )
            
            # Index JSON knowledge graph (kg_llm_navigation)
            kapso.index_kg(
                data_path="benchmarks/mle/data/kg_data.json",
                save_to="data/indexes/kaggle.index",
                search_type="kg_llm_navigation",
            )
        """
        if save_to is None:
            raise ValueError("save_to is required - specify where to save the .index file")
        
        if not wiki_dir and not data_path:
            raise ValueError("Must provide either wiki_dir or data_path")
        
        # Determine search type
        if search_type is None:
            if data_path:
                # JSON data implies kg_llm_navigation
                search_type = "kg_llm_navigation"
            else:
                # Wiki dir implies kg_graph_search (or use config default)
                mode = self._config.get("default_mode", "GENERIC")
                mode_config = self._config.get("modes", {}).get(mode, {})
                search_config = mode_config.get("knowledge_search", {})
                search_type = search_config.get("type", "kg_graph_search")
        
        # Get params from config
        mode = self._config.get("default_mode", "GENERIC")
        mode_config = self._config.get("modes", {}).get(mode, {})
        search_config = mode_config.get("knowledge_search", {})
        params = search_config.get("params", {}).copy()
        
        # Create search backend
        self.knowledge_search = KnowledgeSearchFactory.create(
            search_type=search_type,
            params=params,
        )
        
        # Clear existing data if force=True
        if force:
            print("  Clearing existing index...")
            self.knowledge_search.clear()
        
        # Determine data source and index
        if wiki_dir:
            data_source = str(wiki_dir)
            print(f"  Indexing wiki: {wiki_dir}")
            self.knowledge_search.index(KGIndexInput(wiki_dir=wiki_dir))
        else:
            data_source = str(data_path)
            print(f"  Indexing JSON: {data_path}")
            # Load JSON and index directly (for kg_llm_navigation)
            with open(data_path) as f:
                graph_data = json.load(f)
            self.knowledge_search.index(graph_data)
        
        # Get page count
        page_count = self.knowledge_search.get_indexed_count()
        
        # Build index metadata
        metadata = KGIndexMetadata(
            version="1.0",
            created_at=datetime.now().isoformat(),
            data_source=data_source,
            search_backend=search_type,
            backend_refs=self.knowledge_search.get_backend_refs(),
            page_count=page_count,
        )
        
        # Save index file
        save_path = Path(save_to)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        self._kg_index_path = str(save_path)
        print(f"  Index saved: {save_to} ({page_count} pages)")
        
        return str(save_path)

    # =========================================================================
    # Public Web Research
    # =========================================================================

    def research(
        self,
        objective: str,
        *,
        mode: ResearchMode = ["idea", "implementation"],
        depth: ResearchDepth = "deep",
    ) -> ResearchFindings:
        """
        Do deep public web research for an objective.
        
        Args:
            objective: What you want to research on the public web.
            mode: "idea" | "implementation" | "study" (or list of modes)
            depth: "light" | "deep"
                Maps to OpenAI `reasoning.effort`:
                - light -> "medium"
                - deep  -> "high"
        
        Returns:
            `ResearchFindings` with fluent accessors:
            - .ideas -> List[Source.Idea]
            - .implementations -> List[Source.Implementation]
            - .report -> Source.ResearchReport (if mode="study")
        """
        if self._web_researcher is None:
            self._web_researcher = Researcher()

        return self._web_researcher.research(objective, mode=mode, depth=depth)
    
    def learn(
        self, 
        *sources: Union[Source.Repo, Source.Solution, Source.Idea, Source.Implementation, Source.ResearchReport, ResearchFindings],
        wiki_dir: str = "data/wikis",
        skip_merge: bool = False,
        kg_index: Optional[str] = None,
    ) -> "PipelineResult":
        """
        Learn from one or more knowledge sources.
        
        This ingests knowledge into the Knowledge Graph (KG) via `KnowledgePipeline`.
        
        Supported sources (MVP):
        - `Source.Repo(...)`
        - `Source.Solution(...)`
        - `Source.Idea(...)`, `Source.Implementation(...)`, `Source.ResearchReport(...)`
        - `ResearchFindings` (output of `Kapso.research(...)`)
        
        Args:
            *sources: One or more Source objects.
            wiki_dir: Path to a local wiki directory (e.g., `data/wikis`) used as
                the KG source-of-truth on disk.
                
                Note:
                - URL-based KG targets (e.g. `https://skills.leeroo.com`) are not
                  supported in this code path yet.
            skip_merge: If True, only extract `WikiPage`s (Stage 1) and skip merging
                into the KG backends (Stage 2). This avoids requiring Neo4j/Weaviate.
            
        Example:
            # Learn from repo + web research and merge into local KG
            kapso.learn(
                Source.Repo("https://github.com/user/repo"),
                kapso.research("How to pick LoRA rank?", mode="idea"),
                wiki_dir="data/wikis",
            )
        """
        if not sources:
            raise ValueError("learn() requires at least one source")

        # Backward-compatible handling: if a URL is provided, fall back to the default local wiki dir.
        resolved_wiki_dir = wiki_dir
        if isinstance(wiki_dir, str) and wiki_dir.startswith(("http://", "https://")):
            print(
                f"Warning: URL wiki_dir not supported yet ({wiki_dir}). "
                "Using local wiki_dir='data/wikis' instead."
            )
            resolved_wiki_dir = "data/wikis"

        # Optional: propagate an existing `.index` file path into the merge agent.
        #
        # Why:
        # - The KnowledgeMerger performs create/edit operations via an MCP server.
        # - That MCP server now supports Option A: initializing from KG_INDEX_PATH.
        # - We pass the index path through pipeline->merger so the Claude Code
        #   subprocess can set KG_INDEX_PATH for the MCP server.
        index_path = kg_index or getattr(self, "_kg_index_path", None)
        merger_params = {"kg_index_path": index_path} if index_path else None

        pipeline = KnowledgePipeline(wiki_dir=resolved_wiki_dir, merger_params=merger_params)
        result = pipeline.run(*sources, skip_merge=skip_merge)

        # Keep a small, user-friendly summary.
        print(
            f"Learn complete: sources={result.sources_processed}, "
            f"extracted_pages={result.total_pages_extracted}, "
            f"created={result.created}, edited={result.edited}, "
            f"errors={len(result.errors)}"
        )

        return result
    
    def evolve(
        self,
        goal: str,
        context: Optional[List[Any]] = None,
        constraints: Optional[List[str]] = None,
        output_path: Optional[str] = None,
        initial_repo: Optional[str] = None,
        max_iterations: int = 10,
        # --- Configuration options ---
        mode: Optional[str] = None,
        coding_agent: Optional[str] = None,
        # --- Directory options ---
        eval_dir: Optional[str] = None,
        data_dir: Optional[str] = None,
        # --- Extra context options ---
        additional_context: str = "",
    ) -> SolutionResult:
        """
        Evolve a solution for the given goal.
        
        Uses the Kapso's knowledge (KG) and online experimentation to
        generate robust software.
        
        Args:
            goal: The high-level objective (problem description)
            context: Optional list of Source objects to learn before evolving
            constraints: List of constraints (e.g., ["latency < 50ms"])
            output_path: Where to save the generated code
            initial_repo: Optional starting repository. Accepts:
                - Local path: "/path/to/repo" or "./relative/path"
                - GitHub URL: "https://github.com/owner/repo" (will be cloned)
                - None: Will search for relevant workflow repo in KG
            max_iterations: Maximum experiment iterations (default: 10)
            
            mode: Configuration mode (GENERIC, MINIMAL, etc.)
            coding_agent: Coding agent to use (aider, gemini, claude_code, openhands)
            
            eval_dir: Path to evaluation files (copied to workspace/kapso_evaluation/)
            data_dir: Path to data files (copied to workspace/kapso_datasets/)
            
            additional_context: Extra context appended to the problem prompt.
                This is the intended integration point for research context.
            
        Returns:
            SolutionResult with code_path, experiment_logs, and metadata
        """
        print(f"\n{'='*60}")
        print(f"EVOLVING: {goal}")
        print(f"{'='*60}")
        print(f"  Max iterations: {max_iterations}")
        print(f"  Coding agent: {coding_agent or 'from config'}")
        if eval_dir:
            print(f"  Eval dir: {eval_dir}")
        if data_dir:
            print(f"  Data dir: {data_dir}")
        
        # Resolve initial_repo: handle URLs, local paths, or workflow search
        resolved_repo = self._resolve_initial_repo(initial_repo, goal)
        if resolved_repo:
            print(f"  Initial repo: {resolved_repo}")
        print()
        
        # Build problem description
        problem = self._build_problem_description(goal, constraints)

        # Build context string from context items (text, not sources)
        # Context items are converted to strings and appended to additional_context
        context_parts = []
        if context:
            for item in context:
                # Convert each context item to string
                context_parts.append(str(item))
        
        # Combine knowledge context + caller-provided context + context items
        #
        # Why:
        # - The system already uses `additional_context` to inject KG snippets.
        # - Research ideas should be injected the same way.
        user_context = (additional_context or "").strip()
        items_context = "\n\n".join(context_parts).strip()
        combined_context = "\n\n".join([c for c in [user_context, items_context] if c])
        
        # Create problem handler with all options
        handler = GenericProblemHandler(
            problem_description=problem,
            eval_dir=eval_dir,
            data_dir=data_dir,
            additional_context=combined_context,
        )
        
        # Create orchestrator
        orchestrator = OrchestratorAgent(
            handler,
            config_path=self.config_path,
            mode=mode,
            coding_agent=coding_agent,
            is_kg_active=self.knowledge_search.is_enabled(),
            knowledge_search=self.knowledge_search if self.knowledge_search.is_enabled() else None,
            # IMPORTANT:
            # - Many callers (CLI + E2E tests) pass `output_path` expecting the final repo to live there.
            # - The orchestration layer owns the experiment workspace (a git repo with branches).
            # - Therefore, when `output_path` is provided, we must use it as the workspace directory
            #   so `solution.code_path` points at a real git repo (with `.kapso/repo_memory.json`).
            workspace_dir=output_path,
            initial_repo=resolved_repo,
            eval_dir=eval_dir,
            data_dir=data_dir,
            goal=goal,
        )
        
        # Run experimentation
        print("Running experiments...")
        solve_result = orchestrator.solve(experiment_max_iter=max_iterations)
        
        # Collect results
        experiment_logs = self._extract_experiment_logs(orchestrator)
        workspace_path = orchestrator.search_strategy.workspace.workspace_dir
        
        # Checkout to best solution
        orchestrator.search_strategy.checkout_to_best_experiment_branch()
        
        # Use custom output path if provided
        code_path = output_path or workspace_path
        
        # Create solution result with final feedback
        solution = SolutionResult(
            goal=goal,
            code_path=code_path,
            experiment_logs=experiment_logs,
            final_feedback=solve_result.final_feedback,
            metadata={
                "constraints": constraints or [],
                "iterations": solve_result.iterations_run,
                "cost": f"${solve_result.total_cost:.3f}",
                "stopped_reason": solve_result.stopped_reason,
            }
        )
        
        print(f"\n{'='*60}")
        print("Evolution Complete")
        print(f"{'='*60}")
        print(f"Solution at: {code_path}")
        print(f"Experiments run: {solve_result.iterations_run}")
        print(f"Total cost: ${solve_result.total_cost:.3f}")
        print(f"Stopped reason: {solve_result.stopped_reason}")
        print(f"Goal achieved: {solution.succeeded}")
        if solution.final_score is not None:
            print(f"Final score: {solution.final_score}")
        
        return solution
    
    def deploy(
        self,
        solution: SolutionResult,
        strategy: DeployStrategy = DeployStrategy.AUTO,
        env_vars: Optional[Dict[str, str]] = None,
        coding_agent: str = "claude_code",
    ) -> Software:
        """
        Deploy a solution to create running software.
        
        Uses the deployment pipeline:
        1. Selector: Analyzes solution and selects strategy (if AUTO)
        2. Adapter: Adapts and deploys via coding agent
        3. Runner: Creates execution backend
        
        Args:
            solution: The SolutionResult from evolve()
            strategy: Where to deploy (AUTO, LOCAL, DOCKER, MODAL, BENTOML)
                - AUTO: System analyzes code and chooses best strategy
                - LOCAL: Run as local Python process (fastest)
                - DOCKER: Run in Docker container (isolated)
                - MODAL: Deploy to Modal.com (serverless, GPU)
                - BENTOML: Deploy with BentoML (production ML)
            env_vars: Environment variables to pass to the software
            coding_agent: Which coding agent for adaptation
            
        Returns:
            Software instance with unified interface:
            - .run(inputs) -> {"status": "success", "output": ...}
            - .stop() -> cleanup resources
            - .logs() -> execution logs
            - .is_healthy() -> health check
            
        Example:
            solution = kapso.evolve(goal="Create a trading bot")
            software = kapso.deploy(solution, strategy=DeployStrategy.LOCAL)
            result = software.run({"ticker": "AAPL"})
            software.stop()
        """
        print(f"\n{'='*60}")
        print(f"DEPLOYING: {solution.goal}")
        print(f"{'='*60}")
        print(f"  Strategy: {strategy}")
        print(f"  Code path: {solution.code_path}")
        print()
        
        config = DeployConfig(
            solution=solution,
            env_vars=env_vars,
            coding_agent=coding_agent,
        )
        
        return DeploymentFactory.create(strategy, config)
    
    # =========================================================================
    # INITIAL REPO RESOLUTION HELPERS
    # =========================================================================
    
    def _resolve_initial_repo(self, initial_repo: Optional[str], goal: str) -> Optional[str]:
        """
        Resolve initial_repo to a local path.
        
        Handles three cases:
        1. GitHub URL: Clone to temp directory
        2. Local path: Use as-is
        3. None: Search for workflow repo in KG
        
        Args:
            initial_repo: Local path, GitHub URL, or None
            goal: The goal (used for workflow search if initial_repo is None)
            
        Returns:
            Local path to repo, or None if no repo found/provided
        """
        if initial_repo is not None:
            # Check if it's a GitHub URL
            if self._is_github_url(initial_repo):
                return self._clone_github_repo(initial_repo)
            # Assume local path
            return initial_repo
        
        # No initial_repo provided - search for workflow repo
        return self._search_workflow_repo(goal)
    
    def _is_github_url(self, path: str) -> bool:
        """Check if path is a GitHub URL."""
        return (
            path.startswith("https://github.com/") or 
            path.startswith("git@github.com:") or
            path.startswith("http://github.com/")
        )
    
    def _clone_github_repo(self, url: str) -> str:
        """
        Clone a GitHub repository to a temporary directory.
        
        Args:
            url: GitHub repository URL
            
        Returns:
            Local path to cloned repository
        """
        import tempfile
        import git
        
        # Create temp directory with meaningful prefix
        temp_dir = tempfile.mkdtemp(prefix="kapso_repo_")
        
        print(f"  Cloning {url}...")
        try:
            git.Repo.clone_from(url, temp_dir)
            print(f"  Cloned to: {temp_dir}")
            return temp_dir
        except Exception as e:
            print(f"  Warning: Failed to clone {url}: {e}")
            # Clean up temp dir on failure
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
            return None
    
    def _search_workflow_repo(self, goal: str) -> Optional[str]:
        """
        Search for a relevant workflow repository in the Knowledge Graph.
        
        Args:
            goal: The goal to search for
            
        Returns:
            Local path to cloned workflow repo, or None if not found
        """
        # Only search if KG is enabled
        if not self.knowledge_search.is_enabled():
            print("  No KG index - skipping workflow search")
            return None
        
        try:
            from kapso.knowledge_base.search.workflow_search import WorkflowRepoSearch
            
            print("  Searching for relevant workflow...")
            workflow_search = WorkflowRepoSearch(kg_search=self.knowledge_search)
            result = workflow_search.search(goal, top_k=1)
            
            if not result.is_empty and result.top_result.github_url:
                starter_url = result.top_result.github_url
                print(f"  Found workflow repo: {starter_url}")
                return self._clone_github_repo(starter_url)
            else:
                print("  No matching workflow found")
                return None
        except Exception as e:
            print(f"  Warning: Workflow search failed: {e}")
            return None
    
    def _build_problem_description(
        self, 
        goal: str, 
        constraints: Optional[List[str]]
    ) -> str:
        """Build the full problem description for the orchestrator."""
        parts = [f"# Goal\n{goal}"]
        
        if constraints:
            parts.append("\n# Constraints")
            for c in constraints:
                parts.append(f"- {c}")
        
        return "\n".join(parts)
    
    def _extract_experiment_logs(self, orchestrator: OrchestratorAgent) -> List[str]:
        """Extract experiment history as string logs."""
        logs = []
        history = orchestrator.search_strategy.get_experiment_history()
        
        for exp in history:
            if hasattr(exp, 'had_error') and exp.had_error:
                logs.append(f"Failed: {exp.solution[:100]}... (Error: {exp.error_message})")
            else:
                score = getattr(exp, 'score', 'N/A')
                logs.append(f"Success: {exp.solution[:100]}... (Score: {score})")
        
        return logs


# =============================================================================
# CONVENIENCE EXPORTS
# =============================================================================

__all__ = [
    "Kapso",
    "KGIndexError",
    "Source",
    "SolutionResult",
    "Software",
    "DeployStrategy",
    "DeployConfig",
    "DeploymentFactory",
    "ResearchFindings",
]
