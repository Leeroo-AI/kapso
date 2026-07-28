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
from typing import Any, Dict, List, Mapping, Optional, Union

from kapso.execution.solution import SolutionResult
from kapso.cross_run.launch.contracts import LaunchTaskContextRequest
from kapso.cross_run.launch.production_evolution import (
    execute_production_evolution,
)
from kapso.knowledge_base.search import KnowledgeSearchFactory, KGIndexInput
from kapso.knowledge_base.search.base import KGIndexMetadata
from kapso.knowledge_base.learners import Source, KnowledgePipeline
from kapso.researcher import Researcher, ResearchDepth, ResearchMode
from kapso.knowledge_base.types import ResearchFindings
from kapso.core.config import load_config, load_effective_config


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
            output_path="./campaign",
            task_context_request=task_context,
            starting_artifact_sources=artifacts,
            dependency_runtime_contract=runtime,
            budget_fidelity_envelope=budget,
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
        params.setdefault("models", mode_config.get("models"))
        params.setdefault("retry", mode_config.get("retry"))

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

        print(
            f"  Knowledge Graph: Loaded ({metadata.page_count} pages from {metadata.search_backend})"
        )

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
            raise ValueError(
                "save_to is required - specify where to save the .index file"
            )

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
        params.setdefault("models", mode_config.get("models"))
        params.setdefault("retry", mode_config.get("retry"))

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
        with open(save_path, "w") as f:
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
            configured_mode = self._config.get("default_mode", "GENERIC")
            mode_config = self._config.get("modes", {}).get(configured_mode, {})
            self._web_researcher = Researcher(
                models=mode_config.get("models"),
                retry_policy=mode_config.get("retry"),
            )

        return self._web_researcher.research(objective, mode=mode, depth=depth)

    def learn(
        self,
        *sources: Union[
            Source.Repo,
            Source.Solution,
            Source.Idea,
            Source.Implementation,
            Source.ResearchReport,
            ResearchFindings,
        ],
        wiki_dir: str = "data/wikis",
        skip_merge: bool = False,
        kg_index: Optional[str] = None,
        github_org: Optional[str] = None,
        is_private: bool = True,
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
            github_org: Optional GitHub organization to push workflow repos to.
                If not provided, repos are created under the authenticated user's account.
            is_private: Whether to create private repos (default: True).
                Set to False to create public repos.

        Example:
            # Learn from repo + web research and merge into local KG
            kapso.learn(
                Source.Repo("https://github.com/user/repo"),
                kapso.research("How to pick LoRA rank?", mode="idea"),
                wiki_dir="data/wikis",
            )

            # Learn and push workflow repos to an organization as public
            kapso.learn(
                Source.Repo("https://github.com/user/repo"),
                github_org="my-org",
                is_private=False,
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
        merger_params = {"kg_index_path": index_path} if index_path else {}

        # Get learner config from mode config
        # This allows config.yaml to specify ingestor/merger params (auth_mode, aws_region, etc.)
        mode = self._config.get("default_mode", "GENERIC")
        mode_config = self._config.get("modes", {}).get(mode, {})
        learner_config = mode_config.get("learner", {})

        # Extract ingestor and merger params from config
        ingestor_params = learner_config.get("ingestor", {}).copy()
        config_merger_params = learner_config.get("merger", {})

        # Override ingestor params with user-provided GitHub settings
        # These take precedence over config.yaml values
        if github_org is not None:
            ingestor_params["github_org"] = github_org
        # is_private overrides github_repo_visibility from config
        # Convert is_private (bool) to visibility string for backward compatibility
        ingestor_params["github_repo_visibility"] = (
            "private" if is_private else "public"
        )

        # Merge config merger params with kg_index_path (kg_index_path takes precedence)
        final_merger_params = {**config_merger_params, **merger_params}

        pipeline = KnowledgePipeline(
            wiki_dir=resolved_wiki_dir,
            ingestor_params=ingestor_params,
            merger_params=final_merger_params,
        )
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
        output_path: str,
        task_context_request: LaunchTaskContextRequest | None = None,
        starting_artifact_sources: (
            Mapping[
                str,
                tuple[str | Path, str],
            ]
            | None
        ) = None,
        dependency_runtime_contract: Mapping[str, Any] | None = None,
        budget_fidelity_envelope: Mapping[str, Any] | None = None,
        config_path: str | None = None,
        scope_id: str | None = None,
        task_family_id: str | None = None,
        task_adapter_id: str | None = None,
        mode: Optional[str] = None,
        coding_agent: Optional[str] = None,
        objective_direction: str = "maximize",
        additional_context: str = "",
        resume: bool = False,
        empty_scope_bootstrap_authorization_id: str | None = None,
    ) -> SolutionResult:
        """Run the sole GitHub-backed launch, retrieval, and edit path."""

        run_root = Path(output_path).expanduser().resolve(strict=False)
        sources = {
            artifact_ref: (
                Path(source).expanduser().resolve(strict=True),
                mount_path,
            )
            for artifact_ref, (source, mount_path) in (
                {} if starting_artifact_sources is None else starting_artifact_sources
            ).items()
        }
        result = execute_production_evolution(
            effective_config=load_effective_config(
                config_path or self.config_path,
                mode,
            ),
            goal=goal,
            run_root=run_root,
            state_root=run_root.parent,
            task_context_request=task_context_request,
            starting_artifact_sources=sources,
            dependency_runtime_contract=dependency_runtime_contract,
            budget_fidelity_envelope=budget_fidelity_envelope,
            scope_id=scope_id,
            task_family_id=task_family_id,
            task_adapter_id=task_adapter_id,
            requested_coding_agent=coding_agent,
            objective_direction=objective_direction,
            additional_context=additional_context,
            resume=resume,
            empty_scope_bootstrap_authorization_id=(
                empty_scope_bootstrap_authorization_id
            ),
        )
        action_result = result.metadata["action_result"]
        return SolutionResult(
            goal=goal,
            code_path=str(result.code_path),
            experiment_logs=[action_result["implementation_summary"]],
            metadata=dict(result.metadata),
        )

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
            strategy: Where to deploy (AUTO, LOCAL, MODAL, BENTOML)
                - AUTO: System analyzes code and chooses best strategy
                - LOCAL: Run as local Python process (fastest)
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
