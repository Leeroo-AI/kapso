# Orchestrator Agent
#
# Main orchestrator that coordinates the experimentation loop.
# Uses pluggable search strategies and knowledge retrievers.
#
# In the new design:
# - Developer agent builds evaluation in kapso_evaluation/
# - Developer agent runs evaluation and reports results
# - FeedbackGenerator decides when to stop
# - ExperimentHistoryStore persists experiment results for MCP access
# - Experiment history is accessed via MCP tools (not context managers)

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from kapso.knowledge_base.search import (
    KnowledgeSearch,
    KnowledgeSearchFactory,
)
from kapso.execution.search_strategies import (
    SearchStrategy,
    SearchStrategyFactory,
)
from kapso.execution.coding_agents.factory import CodingAgentFactory
from kapso.execution.search_strategies.generic import FeedbackGenerator, FeedbackResult
from kapso.environment.handlers.base import ProblemHandler
from kapso.core.llm import LLMBackend
from kapso.core.config import load_mode_config
from kapso.execution.search_strategies.base import ExperimentResult, SearchNode
from kapso.execution.memories.experiment_memory import ExperimentHistoryStore
from kapso.execution.iteration_evaluator import (
    IterationEvaluationContext,
    IterationEvaluationError,
    IterationEvaluator,
    normalize_failure_policy,
    normalize_result,
)
from kapso.execution.evaluation_integrity import (
    build_evaluation_manifest,
    manifest_fingerprint,
    verify_data_manifest,
)
from kapso.execution.run_checkpoint import (
    RunCheckpoint,
    RunCheckpointError,
    RunCheckpointCorruptError,
    RunCheckpointIncompatibleError,
    RunCheckpointStore,
    config_fingerprint,
)
from kapso.execution.budget import (
    BudgetLedger,
    BudgetSnapshot,
    BudgetSpec,
    CostEntry,
)
from kapso.execution.evaluation_maintainer import (
    EvaluationChangeRequest,
    EvaluationMaintainer,
    EvaluationMaintainerError,
)
from kapso.execution.fidelity import (
    FULL_PASSTHROUGH,
    ComparabilityClass,
    FidelityPolicy,
    FidelitySpec,
)


CHANGE_REQUEST_PATTERN = re.compile(
    r"<evaluation_change_request>(.*?)</evaluation_change_request>",
    re.DOTALL,
)

# The fast fraction is deliberately absent: it is single-sourced in the
# fidelity block (budget.fidelity.eval.fast_fraction).
_MAINTAINER_BLOCK_KEYS = {
    "type",
    "model",
    "debug_model",
    "agent_specific",
    "subsample_seed",
    "calibration_fraction",
    "calibration_timeout_seconds",
    "fast_variant_threshold_minutes",
    "overhead_factor",
    "max_change_requests",
    "protected_data_paths",
}


@dataclass
class SolveResult:
    """Result from orchestrator.solve()."""
    best_experiment: Optional[ExperimentResult]
    final_feedback: Optional[FeedbackResult]
    stopped_reason: str  # "goal_achieved", "max_iterations", "budget_exhausted", "legacy_stop"
    iterations_run: int
    total_cost: float
    cumulative_iterations: int = 0
    # Which budget dial triggered a budget_exhausted stop:
    # "time_budget" | "cost_budget" | "finalization_reserve" | None
    stop_detail: Optional[str] = None


class OrchestratorAgent:
    """
    Main orchestrator agent that coordinates the experimentation loop.
    
    Uses pluggable components:
    - Search strategies (registered via @register_strategy decorator)
    - Knowledge search backends (registered via @register_knowledge_search decorator)
    - Coding agents (Aider, Gemini, Claude Code, OpenHands)
    - Feedback generator (LLM-based, decides when to stop)
    - Experiment history store (persists results for MCP access)
    
    Args:
        problem_handler: Handler for the problem being solved
        config_path: Path to benchmark-specific config.yaml file
        mode: Configuration mode to use (if None, uses default_mode from config)
        coding_agent: Coding agent to use (overrides config if specified)
        is_kg_active: Whether to use the knowledge graph
        goal: The goal/objective for the evolve process
        iteration_evaluator: Optional observational callback for each finalized
            candidate Git ref
        iteration_evaluator_failure_policy: ``record`` or ``raise``
    """
    
    def __init__(
        self, 
        problem_handler: ProblemHandler,
        config_path: Optional[str] = None,
        mode: Optional[str] = None,
        coding_agent: Optional[str] = None,
        is_kg_active: bool = False,
        knowledge_search: Optional[KnowledgeSearch] = None,
        workspace_dir: Optional[str] = None,
        resume: bool = False,
        iteration_evaluator: Optional[IterationEvaluator] = None,
        iteration_evaluator_failure_policy: str = "record",
        initial_repo: Optional[str] = None,
        eval_dir: Optional[str] = None,
        data_dir: Optional[str] = None,
        goal: Optional[str] = None,
    ):
        self.problem_handler = problem_handler
        self.config_path = config_path
        self.mode = mode
        self.goal = goal or ""
        # Load once before constructing shared services so model roles and retry
        # behavior apply consistently across strategy, memory, and commit calls.
        self.mode_config = load_mode_config(config_path, mode)
        model_routes = self.mode_config.get("models")
        retry_config = self.mode_config.get("retry")
        if model_routes is None and retry_config is None:
            self.llm = LLMBackend()
        else:
            self.llm = LLMBackend(
                models=model_routes,
                retry_policy=retry_config,
            )
        # Optional: seed experiments from an existing local repo (copy/clone into workspace).
        self.initial_repo = initial_repo
        # Optional: directories to copy into workspace
        self.eval_dir = eval_dir
        self.data_dir = data_dir
        self.resume = resume
        self.iteration_evaluator = iteration_evaluator
        self.iteration_evaluator_failure_policy = normalize_failure_policy(
            iteration_evaluator_failure_policy
        )
        
        (
            self.strategy_type,
            self.strategy_params,
        ) = self._resolve_search_strategy_config()
        # Budgets are operator dials, not campaign identity — like
        # max_iterations (a solve() argument that was never fingerprinted).
        # Excluding the budget block keeps "resume with a bigger budget"
        # possible under strict resume validation.
        self._config_budget = dict(self.mode_config.get("budget") or {})
        fingerprint_config = {
            "strategy_type": self.strategy_type,
            "strategy_params": self.strategy_params,
            "mode": self.mode,
            "mode_config": {
                key: value
                for key, value in self.mode_config.items()
                if key != "budget"
            },
            "coding_agent_override": coding_agent,
        }
        self._provided_evaluation_manifest: Optional[Dict[str, str]] = None
        if eval_dir:
            self._provided_evaluation_manifest = build_evaluation_manifest(
                eval_dir
            )
            fingerprint_config["provided_evaluation_fingerprint"] = (
                manifest_fingerprint(self._provided_evaluation_manifest)
            )
        if self.iteration_evaluator is not None:
            fingerprint_config["iteration_evaluator"] = (
                self._callable_identity(self.iteration_evaluator)
            )
            fingerprint_config["iteration_evaluator_failure_policy"] = (
                self.iteration_evaluator_failure_policy
            )
        self.config_fingerprint = config_fingerprint(fingerprint_config)
        
        # Determine workspace directory for experiment history
        self._workspace_dir = workspace_dir

        self.checkpoint_store: Optional[RunCheckpointStore] = None
        self._resume_checkpoint: Optional[RunCheckpoint] = None
        self.completed_iterations = 0
        self._prior_cost = 0.0
        self._prior_elapsed_seconds = 0.0
        self._prior_cost_by_component: Dict[str, float] = {}
        self._restored_node_count = 0

        if workspace_dir is not None:
            self.checkpoint_store = RunCheckpointStore(workspace_dir)

        if self.resume:
            if workspace_dir is None:
                raise ValueError(
                    "Resuming an evolution campaign requires workspace_dir"
                )
            if self.checkpoint_store is None:
                raise AssertionError("Checkpoint store was not initialized")

            checkpoint = self.checkpoint_store.load()
            checkpoint.validate_resume(
                goal=self.goal,
                strategy_type=self.strategy_type,
                config_fingerprint=self.config_fingerprint,
            )
            self._resume_checkpoint = checkpoint
            self.completed_iterations = checkpoint.completed_iterations
            self._prior_cost = float(checkpoint.cumulative_cost)
            self._prior_elapsed_seconds = float(checkpoint.elapsed_seconds)
            self._prior_cost_by_component = dict(
                checkpoint.cost_by_component
            )
        elif self.checkpoint_store is not None and self.checkpoint_store.exists():
            raise RunCheckpointIncompatibleError(
                "This workspace already contains a run checkpoint; pass "
                "resume=True or choose a new output path"
            )
        
        # Create experiment history store
        # Path is determined after search strategy creates workspace
        self.experiment_store: Optional[ExperimentHistoryStore] = None
        
        # Create feedback generator FIRST (needed by search strategy)
        self.feedback_generator = self._create_feedback_generator(coding_agent)
        
        # Track feedback for next iteration
        self.current_feedback: Optional[str] = (
            self._resume_checkpoint.current_feedback
            if self._resume_checkpoint is not None
            else None
        )
        self.last_feedback_result: Optional[FeedbackResult] = None
        
        # Create search strategy (uses feedback_generator)
        self.search_strategy = self._create_search_strategy(
            coding_agent=coding_agent,
            workspace_dir=workspace_dir,
            start_from_checkpoint=self.resume,
        )

        if self._resume_checkpoint is not None:
            try:
                self.search_strategy.load_state(
                    self._resume_checkpoint.strategy_state
                )
                self._validate_restored_branch_refs()
            except RunCheckpointError:
                raise
            except Exception as exc:
                raise RunCheckpointCorruptError(
                    "Could not restore search strategy state from the run "
                    "checkpoint"
                ) from exc
            # Nodes restored from the checkpoint already carried their agent
            # spend into cumulative_cost; only nodes created after this point
            # contribute live phase costs.
            self._restored_node_count = len(
                self.search_strategy.get_experiment_history()
            )

        if self.checkpoint_store is None:
            self.checkpoint_store = RunCheckpointStore(
                self.search_strategy.workspace_dir
            )

        self.budget_ledger = self._create_budget_ledger()
        # Config-only view until solve() re-resolves with explicit args.
        self.budget_spec = BudgetSpec.resolve(config_block=self._config_budget)

        # Resolved before the maintainer: the fast fraction is single-sourced
        # in the fidelity block and the maintainer calibrates at that value.
        self.fidelity_spec = FidelitySpec.resolve(
            self._config_budget.get("fidelity")
        )
        (
            self.evaluation_maintainer,
            self._max_change_requests,
        ) = self._create_evaluation_maintainer()
        self._change_requests_filed = 0
        self._fidelity_active = False
        if (
            self.fidelity_spec.mode != "off"
            and self.evaluation_maintainer is None
        ):
            raise ValueError(
                "Fidelity requires an evaluation_maintainer block: the "
                "policy's timing model comes from measured evaluation runs"
            )

        # Now create experiment history store with the actual workspace path
        experiment_history_path = os.path.join(
            self.search_strategy.workspace_dir, 
            ".kapso", 
            "experiment_history.json"
        )
        self.experiment_store = ExperimentHistoryStore(
            json_path=experiment_history_path,
            weaviate_url=os.environ.get("WEAVIATE_URL"),
            goal=self.goal,
            llm=self.llm,
        )
        
        # Create knowledge search backend (or use provided instance).
        # This allows Kapso.evolve() to inject a concrete backend (e.g., kg_graph_search)
        # without relying on config defaults (which may point to a different backend).
        if knowledge_search is not None:
            self.knowledge_search = knowledge_search
            self._owns_knowledge_search = False
        else:
            self.knowledge_search = self._create_knowledge_search(
                is_kg_active=is_kg_active,
            )
            # We created it inside the orchestrator → we should close it.
            self._owns_knowledge_search = True
    
    def _create_feedback_generator(
        self,
        coding_agent: Optional[str] = None,
    ) -> FeedbackGenerator:
        """
        Create feedback generator.
        
        Uses the same coding agent type as the developer agent by default.
        """
        # Get coding agent config from mode config
        mode_config = self.mode_config
        # Check for dedicated feedback_generator config first, fall back to coding_agent
        feedback_config = mode_config.get('feedback_generator', {}) if mode_config else {}
        coding_config = mode_config.get('coding_agent', {}) if mode_config else {}
        
        # Use feedback_generator config if available, otherwise fall back to coding_agent config
        if feedback_config:
            agent_type = feedback_config.get('type', 'claude_code')
            agent_model = feedback_config.get('model')
            agent_debug_model = feedback_config.get('debug_model')
            agent_specific = feedback_config.get('agent_specific', {})
        else:
            # Fall back to coding_agent config
            agent_type = coding_agent or coding_config.get('type', 'claude_code')
            agent_model = coding_config.get('model')
            agent_debug_model = coding_config.get('debug_model')
            agent_specific = coding_config.get('agent_specific', {})
        
        # Build config for feedback generator
        feedback_agent_config = CodingAgentFactory.build_config(
            agent_type=agent_type,
            model=agent_model,
            debug_model=agent_debug_model,
            agent_specific=agent_specific,
        )
        
        return FeedbackGenerator(
            coding_agent_config=feedback_agent_config,
        )

    def _create_search_strategy(
        self,
        coding_agent: Optional[str],
        workspace_dir: Optional[str],
        start_from_checkpoint: bool,
    ) -> SearchStrategy:
        """
        Create search strategy from config.
        
        Args:
            coding_agent: Override coding agent type
            
        Returns:
            Configured SearchStrategy instance
        """
        mode_config = self.mode_config
        strategy_type = self.strategy_type
        strategy_params = self.strategy_params

        if not mode_config:
            coding_agent_type = coding_agent or "claude_code"
            coding_agent_model = None
            coding_agent_debug_model = None
            coding_agent_specific = None
        else:
            # Extract coding agent config
            coding_config = mode_config.get('coding_agent', {})
            if coding_agent:
                coding_agent_type = coding_agent
                coding_agent_model = None
                coding_agent_debug_model = None
                coding_agent_specific = None
            elif coding_config:
                coding_agent_type = coding_config.get('type', 'aider')
                coding_agent_model = coding_config.get('model')
                coding_agent_debug_model = coding_config.get('debug_model')
                # Support agent_specific from YAML config (e.g., auth_mode for Claude Code)
                coding_agent_specific = coding_config.get('agent_specific')
            else:
                coding_agent_type = 'aider'
                coding_agent_model = mode_config.get('developer_model')
                coding_agent_debug_model = mode_config.get('developer_debug_model')
                coding_agent_specific = None
        
        # Build coding agent config
        coding_agent_config = CodingAgentFactory.build_config(
            agent_type=coding_agent_type,
            model=coding_agent_model,
            debug_model=coding_agent_debug_model,
            agent_specific=coding_agent_specific,
        )
        
        # Create strategy via factory
        return SearchStrategyFactory.create(
            strategy_type=strategy_type,
            problem_handler=self.problem_handler,
            llm=self.llm,
            coding_agent_config=coding_agent_config,
            params=strategy_params,
            workspace_dir=workspace_dir,
            start_from_checkpoint=start_from_checkpoint,
            initial_repo=self.initial_repo,
            eval_dir=self.eval_dir,
            evaluation_manifest=self._provided_evaluation_manifest,
            data_dir=self.data_dir,
            feedback_generator=self.feedback_generator,
            goal=self.goal,
        )

    def _resolve_search_strategy_config(self) -> Tuple[str, Dict[str, Any]]:
        """Resolve strategy identity before a resume mutates the workspace."""
        mode_config = self.mode_config
        if not mode_config:
            return "generic", {}

        search_config = mode_config.get("search_strategy", {})
        if search_config:
            return (
                search_config.get("type", "generic"),
                search_config.get("params", {}) or {},
            )

        return (
            "generic",
            {
                "reasoning_effort": mode_config.get(
                    "reasoning_effort", "medium"
                ),
                "code_debug_tries": mode_config.get(
                    "code_debug_tries", 5
                ),
                "node_expansion_limit": mode_config.get(
                    "node_expansion_limit", 2
                ),
                "node_expansion_new_childs_count": mode_config.get(
                    "node_expansion_new_childs_count", 5
                ),
                "idea_generation_steps": mode_config.get(
                    "idea_generation_steps", 1
                ),
                "first_experiment_factor": mode_config.get(
                    "first_experiment_factor", 1
                ),
                "experimentation_per_run": mode_config.get(
                    "experimentation_per_run", 1
                ),
                "per_step_maximum_solution_count": mode_config.get(
                    "per_step_maximum_solution_count", 10
                ),
                "exploration_budget_percent": mode_config.get(
                    "exploration_budget_percent", 30
                ),
                "idea_generation_model": mode_config.get(
                    "idea_generation_model", "reasoning"
                ),
                "idea_generation_ensemble_models": mode_config.get(
                    "idea_generation_ensemble_models", ["reasoning"]
                ),
            },
        )

    def _create_knowledge_search(
        self,
        is_kg_active: bool,
    ) -> KnowledgeSearch:
        """
        Create knowledge search backend from config.
        
        Args:
            is_kg_active: Whether to enable knowledge graph
            
        Returns:
            Configured KnowledgeSearch instance
        """
        mode_config = self.mode_config
        
        # Check for knowledge_search config (new format)
        ks_config = mode_config.get('knowledge_search', {})
        
        if ks_config:
            resolved_config = dict(ks_config)
            resolved_params = dict(resolved_config.get("params") or {})
            resolved_params.setdefault("models", mode_config.get("models"))
            resolved_params.setdefault("retry", mode_config.get("retry"))
            resolved_config["params"] = resolved_params
            return KnowledgeSearchFactory.create_from_config(resolved_config)
        
        # Check for legacy knowledge_retriever config
        kr_config = mode_config.get('knowledge_retriever', {})
        
        if kr_config:
            # Convert legacy config to new format
            resolved_params = dict(kr_config.get("params") or {})
            resolved_params.setdefault("models", mode_config.get("models"))
            resolved_params.setdefault("retry", mode_config.get("retry"))
            return KnowledgeSearchFactory.create_from_config({
                "type": "kg_llm_navigation",
                "enabled": kr_config.get("enabled", True),
                "params": resolved_params,
                "preset": kr_config.get("preset"),
            })
        
        # Check use_knowledge_graph flag
        if 'use_knowledge_graph' in mode_config:
            kg_enabled = mode_config.get('use_knowledge_graph', False)
            if kg_enabled or is_kg_active:
                return KnowledgeSearchFactory.create(
                    search_type="kg_llm_navigation",
                    params={
                        "models": mode_config.get("models"),
                        "retry": mode_config.get("retry"),
                    },
                )
            else:
                return KnowledgeSearchFactory.create_null()
        
        # Fall back to is_kg_active parameter
        if is_kg_active:
            return KnowledgeSearchFactory.create(
                search_type="kg_llm_navigation",
                params={
                    "models": mode_config.get("models"),
                    "retry": mode_config.get("retry"),
                },
            )
        
        # Default: disabled
        return KnowledgeSearchFactory.create_null()

    def _live_phase_costs(self) -> Dict[str, float]:
        """Attributed agent spend from nodes created in this process slice."""
        live: Dict[str, float] = {}
        history = self.search_strategy.get_experiment_history()
        for node in history[self._restored_node_count:]:
            for phase_name, phase_values in getattr(
                node, "phase_telemetry", {}
            ).items():
                live[phase_name] = (
                    live.get(phase_name, 0.0)
                    + phase_values.get("cost_usd", 0.0)
                )
        return live

    def _create_evaluation_maintainer(self):
        """Build the maintainer when the mode config declares one."""
        block = self.mode_config.get("evaluation_maintainer")
        if not block:
            return None, 0
        unknown = sorted(set(block) - _MAINTAINER_BLOCK_KEYS)
        if unknown:
            raise ValueError(
                "Unknown evaluation_maintainer config keys: "
                + ", ".join(unknown)
            )
        agent_config = CodingAgentFactory.build_config(
            agent_type=block.get("type", "claude_code"),
            model=block.get("model"),
            debug_model=block.get("debug_model"),
            agent_specific=block.get("agent_specific", {}),
        )
        maintainer = EvaluationMaintainer(
            coding_agent_config=agent_config,
            workspace_dir=self.search_strategy.workspace_dir,
            # Single-sourced in the fidelity block: the fraction the policy
            # requests is the fraction the maintainer calibrates and registers.
            fast_fraction=self.fidelity_spec.eval_fast_fraction,
            subsample_seed=block.get("subsample_seed", 1337),
            calibration_fraction=block.get("calibration_fraction", 0.03),
            calibration_timeout_seconds=block.get(
                "calibration_timeout_seconds", 900
            ),
            fast_variant_threshold_seconds=(
                block.get("fast_variant_threshold_minutes", 20) * 60
            ),
            overhead_factor=block.get("overhead_factor", 1.25),
            protected_data_paths=block.get("protected_data_paths", []),
        )
        return maintainer, block.get("max_change_requests", 3)

    def _record_maintainer_spend(self) -> None:
        telemetry = self.evaluation_maintainer.last_transaction_telemetry
        if telemetry is not None:
            self.budget_ledger.record(
                CostEntry(
                    component="evaluation_maintenance",
                    cost_usd=telemetry.cost_usd,
                    duration_seconds=telemetry.duration_seconds,
                )
            )

    def _adopt_registered_evaluation(self) -> None:
        """Point the strategy at the registered evaluator head."""
        manifest = build_evaluation_manifest(
            self.evaluation_maintainer.evaluation_dir
        )
        head = self.evaluation_maintainer.registry.head()
        self.search_strategy.set_registered_evaluation(
            manifest=manifest,
            command=self.evaluation_maintainer.evaluation_command(
                fidelity="full", fraction=1.0
            ),
            evaluator_id=manifest_fingerprint(manifest),
            subsample_seed=self.evaluation_maintainer.subsample_seed,
            data_manifest=head.data_manifest,
        )

    def _ensure_evaluation_registered(self) -> None:
        """Run the maintainer's setup once; validate consistency on resume."""
        if self.evaluation_maintainer is None:
            return
        registry = self.evaluation_maintainer.registry
        if registry.exists():
            head = registry.head()
            current = build_evaluation_manifest(
                self.evaluation_maintainer.evaluation_dir
            )
            if manifest_fingerprint(current) != head.evaluator_id:
                raise EvaluationMaintainerError(
                    "Workspace evaluation tree does not match the registered "
                    "evaluator head; the maintainer registry is the only "
                    "sanctioned path for evaluation changes"
                )
            data_problem = verify_data_manifest(
                self.search_strategy.workspace_dir, head.data_manifest
            )
            if data_problem:
                raise EvaluationMaintainerError(
                    "Workspace evaluation inputs do not match the registered "
                    f"head: {data_problem}"
                )
        else:
            self.evaluation_maintainer.setup(
                goal=self.goal,
                eval_dir=self.eval_dir,
                data_dir=self.data_dir,
            )
            self._record_maintainer_spend()
        self._adopt_registered_evaluation()

    def _canonical_evaluation_params(self):
        """(fidelity, fraction) of the class node.score projections use."""
        if self._fidelity_active:
            return "fast", self.fidelity_spec.eval_fast_fraction
        return "full", 1.0

    def _execute_evaluator_transition(
        self, priority_node_id: Optional[int] = None
    ) -> None:
        """Anchor the frontier on the new evaluator head.

        Durable state machine: pending is checkpointed before the bridge
        runs, anchored after — a crash in between replays the bridge
        idempotently on resume. Fallbacks are mechanical: candidates whose
        artifacts are gone or whose bridge fails fall through to the next;
        with none left the frontier is legitimately empty and the next
        iteration drafts from baseline.

        ``priority_node_id`` bridges first: an accepted change request is
        the maintainer certifying that the requester's old measurement was
        unsound, so that node has the strongest claim to a new-ruler
        measurement — its old score (often None, because of the very
        defect just confirmed) must not decide the order. Persisted in the
        pending record so a crash replays the same priority.
        """
        strategy = self.search_strategy
        new_evaluator_id = strategy.registered_evaluator_id
        old_evaluator_id = strategy.scores_evaluator_id
        fidelity, fraction = self._canonical_evaluation_params()
        # Affordability window, not the timing estimate: the bridge is
        # delivery-critical and may legitimately run inside the reserve.
        deadline = (
            max(
                0.0,
                self.budget_spec.time_budget_seconds
                - self.get_elapsed_seconds(),
            )
            if self.budget_spec.time_budget_seconds is not None
            else None
        )

        print(
            "[Orchestrator] Evaluator transition: anchoring the frontier "
            f"on {new_evaluator_id[:12]} (was "
            f"{old_evaluator_id[:12] if old_evaluator_id else '<none>'})"
        )
        strategy.evaluator_transition = {
            "old_evaluator_id": old_evaluator_id,
            "new_evaluator_id": new_evaluator_id,
            "status": "pending",
            **(
                {"priority_node_id": priority_node_id}
                if priority_node_id is not None
                else {}
            ),
        }
        self._save_run_checkpoint(status="running")

        # Tampering (a non-empty integrity error) is a property of the
        # candidate and stays exclusionary. evaluation_valid=False with a
        # clean integrity record means only the OLD measurement was
        # unsound — often because of the very defect this transition
        # fixes — and a fresh measurement under the new head is exactly
        # what the bridge exists to buy. The live CR campaign's requester
        # was filtered out by the old evaluation_valid check and the
        # frontier restarted from baseline for no reason.
        candidates = sorted(
            (
                node
                for node in strategy.get_experiment_history()
                if not node.had_error
                and node.branch_name
                and not node.evaluation_integrity_error
            ),
            key=lambda node: node.score if node.score is not None else float(
                "-inf"
            ),
            reverse=True,
        )
        if priority_node_id is not None:
            candidates = [
                node
                for node in candidates
                if node.node_id == priority_node_id
            ] + [
                node
                for node in candidates
                if node.node_id != priority_node_id
            ]
        bridged = False
        for candidate in candidates:
            bridged = strategy.run_bridge_evaluation(
                candidate,
                fidelity=fidelity,
                fraction=fraction,
                deadline_seconds=deadline,
            )
            if bridged:
                print(
                    "[Orchestrator] Bridge evaluation anchored node "
                    f"{candidate.node_id} under the new evaluator"
                )
                break
        if not bridged:
            print(
                "[Orchestrator] No candidate could bridge to the new "
                "evaluator; the frontier restarts from baseline"
            )

        strategy.refresh_score_projections(
            ComparabilityClass(
                evaluator_id=new_evaluator_id,
                fidelity=fidelity,
                fraction=fraction,
                seed=strategy.registered_subsample_seed,
            )
        )
        strategy.scores_evaluator_id = new_evaluator_id
        strategy.evaluator_transition = {
            "old_evaluator_id": old_evaluator_id,
            "new_evaluator_id": new_evaluator_id,
            "status": "anchored",
            **(
                {"priority_node_id": priority_node_id}
                if priority_node_id is not None
                else {}
            ),
        }
        self._save_run_checkpoint(status="running")

    def _reconcile_evaluator_state(self) -> None:
        """Adopt or replay transitions so scores match the registry head."""
        if self.evaluation_maintainer is None:
            return
        strategy = self.search_strategy
        head_id = strategy.registered_evaluator_id
        transition = strategy.evaluator_transition
        pending = (
            transition is not None and transition.get("status") == "pending"
        )
        if not strategy.get_experiment_history():
            # Nothing measured yet: adopt the head as the scoring ruler.
            strategy.scores_evaluator_id = head_id
            return
        if pending or strategy.scores_evaluator_id != head_id:
            self._execute_evaluator_transition(
                priority_node_id=(
                    transition.get("priority_node_id") if pending else None
                )
            )

    def _probe_estimate_seconds(self, budget_spec: BudgetSpec) -> float:
        """Measured mean probe duration; the iteration floor before data."""
        durations = [
            node.duration_seconds
            for node in self.search_strategy.get_experiment_history()
            if getattr(node, "eval_fidelity", "full") == "fast"
            and node.duration_seconds is not None
        ]
        if durations:
            return sum(durations) / len(durations)
        return budget_spec.min_iteration_seconds

    def _route_change_requests(self, candidates: List[SearchNode]) -> None:
        """File explicit <evaluation_change_request> tags with the maintainer."""
        if self.evaluation_maintainer is None:
            return
        for candidate in candidates:
            match = CHANGE_REQUEST_PATTERN.search(candidate.agent_output or "")
            if match is None:
                continue
            if self._change_requests_filed >= self._max_change_requests:
                print(
                    "[Orchestrator] Change-request cap reached "
                    f"({self._max_change_requests}); not routing further "
                    "requests this campaign"
                )
                return
            self._change_requests_filed += 1
            outcome = self.evaluation_maintainer.handle_change_request(
                EvaluationChangeRequest(
                    iteration=self.completed_iterations + 1,
                    requested_by="implementation",
                    summary=match.group(1).strip(),
                    evidence=candidate.evaluation_output or "",
                )
            )
            self._record_maintainer_spend()
            print(
                "[Orchestrator] Evaluation change request "
                f"{'accepted' if outcome.accepted else 'rejected'}: "
                f"{outcome.reason}"
            )
            if outcome.accepted:
                self._adopt_registered_evaluation()
                print(
                    "[Orchestrator] Evaluator re-registered as "
                    f"v{outcome.new_version.version}"
                )
                self._execute_evaluator_transition(
                    priority_node_id=candidate.node_id
                )

    def _create_budget_ledger(self) -> BudgetLedger:
        """Wire the ledger: priors from the checkpoint, live meters, nodes."""
        ledger = BudgetLedger(
            prior_elapsed_seconds=self._prior_elapsed_seconds,
            prior_cost_usd=self._prior_cost,
            prior_cost_by_component=self._prior_cost_by_component,
        )
        ledger.set_meter("llm_backend", self.llm.get_cumulative_cost)
        ledger.set_meter(
            "workspace_sessions",
            self.search_strategy.workspace.get_cumulative_cost,
        )
        ledger.set_phase_cost_provider(self._live_phase_costs)
        return ledger

    def get_cumulative_cost(self) -> float:
        """Get total cost from all components, attributed agents included."""
        return self.budget_ledger.total_cost()

    def get_elapsed_seconds(self) -> float:
        """The durable clock: prior slices plus the live one."""
        return self.budget_ledger.elapsed_seconds()

    def _save_run_checkpoint(
        self, *, status: str, last_stop: Optional[str] = None
    ) -> None:
        """Atomically persist orchestration and strategy state."""
        if self.checkpoint_store is None:
            self.checkpoint_store = RunCheckpointStore(
                self.search_strategy.workspace_dir
            )
        checkpoint = RunCheckpoint.create(
            strategy_type=self.strategy_type,
            goal=self.goal,
            config_fingerprint=self.config_fingerprint,
            status=status,
            completed_iterations=self.completed_iterations,
            cumulative_cost=self.get_cumulative_cost(),
            current_feedback=self.current_feedback,
            strategy_state=self.search_strategy.dump_state(),
            elapsed_seconds=self.get_elapsed_seconds(),
            cost_by_component=self.budget_ledger.cost_by_component(),
            last_stop=last_stop,
        )
        self.checkpoint_store.save(checkpoint)

    def _validate_restored_branch_refs(self) -> None:
        """Ensure successful checkpoint nodes still point to Git refs."""
        import git

        repo = git.Repo(self.search_strategy.workspace_dir)
        missing = []
        for node in self.search_strategy.get_experiment_history():
            branch_name = getattr(node, "branch_name", "")
            if not branch_name or getattr(node, "had_error", False):
                continue
            try:
                repo.commit(branch_name)
            except (git.BadName, git.GitCommandError, ValueError):
                missing.append(branch_name)
        if missing:
            unique = ", ".join(sorted(set(missing)))
            raise RunCheckpointCorruptError(
                f"Run checkpoint references missing Git branches: {unique}"
            )

    @staticmethod
    def _callable_identity(callback: IterationEvaluator) -> str:
        """Return a stable-enough identity for strict resume compatibility."""
        module = getattr(callback, "__module__", type(callback).__module__)
        qualname = getattr(
            callback,
            "__qualname__",
            type(callback).__qualname__,
        )
        return f"{module}.{qualname}"

    @staticmethod
    def _new_candidates(
        previous_node_ids: set[int],
        history: List[SearchNode],
        returned_node: Optional[SearchNode],
    ) -> List[SearchNode]:
        """Find candidates finalized by the current strategy iteration."""
        candidates = []
        seen_node_ids = set(previous_node_ids)
        for candidate in history:
            if candidate.node_id in seen_node_ids:
                continue
            candidates.append(candidate)
            seen_node_ids.add(candidate.node_id)
        if (
            returned_node is not None
            and returned_node.node_id not in seen_node_ids
        ):
            candidates.append(returned_node)
        return candidates

    def _evaluate_candidates(
        self,
        candidates: List[SearchNode],
        *,
        iteration: int,
    ) -> None:
        """Evaluate each finalized ref in an isolated detached worktree."""
        if self.iteration_evaluator is None:
            return

        for candidate in candidates:
            git_ref = candidate.branch_name
            parent_ref = candidate.parent_branch_name or "main"
            try:
                if not git_ref:
                    raise ValueError(
                        "candidate does not identify a Git branch"
                    )
                with self.search_strategy.workspace.materialize_ref(
                    git_ref
                ) as materialized_dir:
                    node_snapshot = SearchNode.from_dict(candidate.to_dict())
                    node_snapshot.workspace_dir = str(materialized_dir)
                    context = IterationEvaluationContext(
                        iteration=iteration,
                        goal=self.goal,
                        workspace_dir=Path(materialized_dir),
                        git_ref=git_ref,
                        parent_ref=parent_ref,
                        node=node_snapshot,
                    )
                    result = normalize_result(
                        self.iteration_evaluator(context)
                    )
                candidate.metrics = dict(result.metrics)
                candidate.primary_metric = result.primary_metric
                candidate.external_evaluation_metadata = dict(
                    result.metadata
                )
                candidate.external_evaluation_error = ""
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                if self.iteration_evaluator_failure_policy == "raise":
                    raise IterationEvaluationError(
                        "Iteration evaluator failed for candidate "
                        f"{candidate.node_id} at "
                        f"{git_ref or '<missing-ref>'}: "
                        f"{exc}"
                    ) from exc
                candidate.metrics = {}
                candidate.primary_metric = None
                candidate.external_evaluation_metadata = {}
                candidate.external_evaluation_error = message
                print(
                    "[Orchestrator] Warning: external evaluation failed for "
                    f"candidate {candidate.node_id}: {message}"
                )

    def solve(
        self,
        experiment_max_iter: int = 20,
        time_budget_minutes: Optional[float] = None,
        cost_budget: Optional[float] = None,
        finalization_reserve_minutes: Optional[float] = None,
    ) -> SolveResult:
        """
        Run the main experimentation loop.
        
        In the new design:
        1. Developer agent implements solution and runs evaluation
        2. Feedback generator validates evaluation and decides stop/continue
        3. Optional external evaluator records observational candidate metrics
        4. Loop continues until goal reached or budget exhausted
        5. Experiment history is accessed via MCP tools (not context managers)
        
        Stops when ANY of these conditions is met:
        1. Feedback generator says STOP (goal achieved)
        2. Budget exhausted (time/cost/iterations)
        3. Legacy: problem_handler.stop_condition() (for backward compatibility)
        
        Args:
            experiment_max_iter: Maximum number of experiment iterations
            time_budget_minutes: Time budget in minutes (optional, no limit if not set)
            cost_budget: Maximum cost in dollars (optional, no limit if not set)
            
        Returns:
            SolveResult with best_experiment, final_feedback, stopped_reason
        """
        # Explicit arguments win over the mode config's optional budget block.
        budget_spec = BudgetSpec.resolve(
            config_block=self._config_budget,
            time_budget_minutes=time_budget_minutes,
            cost_budget=cost_budget,
            finalization_reserve_minutes=finalization_reserve_minutes,
        )
        # Pinned for components that run outside the iteration loop (the
        # evaluator-transition bridge derives its deadline from it).
        self.budget_spec = budget_spec
        self.budget_ledger.start_clock()
        # The maintainer's setup transaction runs inside the budgeted clock,
        # before iteration 1; on resume it validates registry consistency.
        self._ensure_evaluation_registered()

        # The fidelity policy: deterministic profile grants over measured
        # evaluation timing. Enabled by mode (or auto-affordability); the
        # escrowed reserve becomes the committed-run slot when active.
        fidelity_policy: Optional[FidelityPolicy] = None
        fidelity_active = False
        if (
            self.fidelity_spec.mode != "off"
            and self.evaluation_maintainer is not None
        ):
            fidelity_policy = FidelityPolicy(
                spec=self.fidelity_spec,
                strategy=self.search_strategy,
                maintainer=self.evaluation_maintainer,
            )
            fidelity_active = fidelity_policy.enabled(
                budget_spec.time_budget_seconds
            )
        self._fidelity_active = fidelity_active
        # Replay a pending transition or anchor restored scores on the
        # current registry head before any selection happens.
        self._reconcile_evaluator_state()
        reserve_run_pending = False

        stopped_reason = "max_iterations"  # default
        stop_detail: Optional[str] = None
        iterations_run = 0

        # Bootstrap checkpoint: registration and reconciliation are paid,
        # durable work — a crash at any later point must resume instead of
        # restarting the campaign (and re-buying the maintainer setup).
        self._save_run_checkpoint(status="running")

        # Get problem context once (experiment history is accessed via MCP)
        problem = self.problem_handler.get_problem_context()

        try:
            for i in range(experiment_max_iter):
                # The escrow is re-derived from history each round: once a
                # full-measured champion exists, the reserve shrinks to the
                # contingency residual and the freed time flows back into
                # the searchable window. Resume-deterministic, like every
                # other policy input.
                effective_reserve_seconds = (
                    fidelity_policy.effective_reserve_seconds(
                        budget_spec.time_budget_seconds,
                        self.search_strategy.get_experiment_history(),
                    )
                    if fidelity_active
                    and budget_spec.time_budget_seconds is not None
                    else budget_spec.finalization_reserve_seconds
                )
                # Build the per-iteration budget view from the durable clock,
                # so a resumed campaign continues its budget instead of
                # resetting it. Strategies get the snapshot read-only.
                snapshot = BudgetSnapshot(
                    iteration_index=i,
                    max_iterations=experiment_max_iter,
                    elapsed_seconds=self.get_elapsed_seconds(),
                    cost_usd=self.get_cumulative_cost(),
                    time_budget_seconds=budget_spec.time_budget_seconds,
                    cost_budget_usd=budget_spec.cost_budget_usd,
                    finalization_reserve_seconds=effective_reserve_seconds,
                    min_agent_timeout_seconds=(
                        budget_spec.min_agent_timeout_seconds
                    ),
                )
                self.search_strategy.observe_budget(snapshot)
                budget_progress = snapshot.progress_percent

                # Check budget exhaustion. A budget stop is a pause, not a
                # completion: the checkpoint stays resumable and records why
                # in last_stop. Only goal achievement completes a campaign.
                if snapshot.exhausted:
                    print("[Orchestrator] Stopping: budget exhausted")
                    stopped_reason = "budget_exhausted"
                    stop_detail = (
                        "time_budget"
                        if snapshot.time_fraction >= 1.0
                        else "cost_budget"
                        if snapshot.cost_fraction >= 1.0
                        else None
                    )
                    self._save_run_checkpoint(
                        status="running",
                        last_stop=stop_detail,
                    )
                    break

                # The reserve gate: refuse admission when what remains
                # outside the escrowed finalization reserve cannot hold one
                # more iteration. Hard arithmetic only — no estimation. With
                # fidelity active and no full-size result yet, the trip does
                # not merely stop: it grants the guaranteed reserve run.
                remaining_after_reserve = snapshot.remaining_after_reserve
                if (
                    remaining_after_reserve is not None
                    and remaining_after_reserve
                    <= budget_spec.min_iteration_seconds
                ):
                    champion = (
                        fidelity_policy.full_champion(
                            self.search_strategy.get_experiment_history()
                        )
                        if fidelity_active
                        else None
                    )
                    if fidelity_active and champion is None:
                        print(
                            "[Orchestrator] Reserve gate: executing the "
                            "escrowed full-size attempt before stopping"
                        )
                        reserve_run_pending = True
                        # The reserve run SPENDS the escrow — except the
                        # measurement's slice. Its snapshot releases the
                        # campaign reserve (the live escrowed iteration was
                        # once killed at the 60s floor with 18 escrowed
                        # minutes on the clock) but keeps the timing
                        # model's full-eval upper as the residual reserve,
                        # so the build cannot starve the frame measurement
                        # that follows it (also observed live: the build
                        # spent the whole escrow and the measurement got
                        # 64 seconds).
                        measurement_slice = (
                            self.evaluation_maintainer.timing(
                                1.0
                            ).upper_seconds
                        )
                        snapshot = BudgetSnapshot(
                            iteration_index=i,
                            max_iterations=experiment_max_iter,
                            elapsed_seconds=self.get_elapsed_seconds(),
                            cost_usd=self.get_cumulative_cost(),
                            time_budget_seconds=(
                                budget_spec.time_budget_seconds
                            ),
                            cost_budget_usd=budget_spec.cost_budget_usd,
                            finalization_reserve_seconds=measurement_slice,
                            min_agent_timeout_seconds=(
                                budget_spec.min_agent_timeout_seconds
                            ),
                        )
                        self.search_strategy.observe_budget(snapshot)
                    else:
                        print(
                            "[Orchestrator] Stopping: finalization reserve "
                            "reached — protecting the endgame window"
                        )
                        stopped_reason = "budget_exhausted"
                        stop_detail = "finalization_reserve"
                        self._save_run_checkpoint(
                            status="running",
                            last_stop=stop_detail,
                        )
                        break

                if fidelity_active:
                    decision = fidelity_policy.decide(
                        nodes=self.search_strategy.get_experiment_history(),
                        remaining_after_reserve=(
                            snapshot.remaining_after_reserve
                        ),
                        probe_estimate_seconds=(
                            self._probe_estimate_seconds(budget_spec)
                        ),
                        reserve_run=reserve_run_pending,
                    )
                else:
                    decision = FULL_PASSTHROUGH
                self.search_strategy.observe_fidelity(decision)

                iterations_run = i + 1
                
                # Build context with problem and feedback
                # Experiment history is accessed via MCP tools by the agent
                context = problem
                if self.current_feedback:
                    context = f"{problem}\n\n## Feedback from Previous Iteration\n\n{self.current_feedback}\n\nPlease address the above feedback in this iteration."
                
                # Run one iteration of search strategy
                # Search strategy handles: solution generation, implementation, feedback
                # Returns SearchNode with all data including should_stop
                previous_node_ids = {
                    candidate.node_id
                    for candidate in (
                        self.search_strategy.get_experiment_history()
                    )
                }
                node = self.search_strategy.run(
                    context, 
                    budget_progress=budget_progress
                )
                
                # Skip if no result (shouldn't happen but be safe)
                if node is None:
                    print(f"[Orchestrator] Warning: No result from iteration {i+1}")
                    continue

                finalized_candidates = self._new_candidates(
                    previous_node_ids,
                    self.search_strategy.get_experiment_history(),
                    node,
                )
                enforce_integrity = getattr(
                    self.search_strategy,
                    "enforce_evaluation_integrity",
                    None,
                )
                if callable(enforce_integrity):
                    for candidate in finalized_candidates:
                        if not getattr(
                            candidate,
                            "_evaluation_integrity_checked",
                            False,
                        ):
                            enforce_integrity(candidate)
                self._evaluate_candidates(
                    finalized_candidates,
                    iteration=self.completed_iterations + 1,
                )

                # Persist every candidate finalized in this strategy iteration.
                # External metrics are attached before history and checkpoint
                # writes so all durable representations agree.
                if self.experiment_store:
                    for candidate in finalized_candidates:
                        self.experiment_store.add_experiment(candidate)

                self._route_change_requests(finalized_candidates)
                
                # Log result
                print(f"[Orchestrator] Iteration {i+1} result:")
                print(f"  - Score: {node.score}")
                print(f"  - Should stop: {node.should_stop}")
                print(f"  - Evaluation valid: {node.evaluation_valid}")
                print(f"  - Feedback: {node.feedback or ''}")
                
                # Store feedback result for return value
                if node.feedback:
                    from kapso.execution.search_strategies.generic import FeedbackResult
                    self.last_feedback_result = FeedbackResult(
                        stop=node.should_stop,
                        evaluation_valid=node.evaluation_valid,
                        feedback=node.feedback,
                        score=node.score,
                    )

                self.current_feedback = node.feedback
                self.completed_iterations += 1
                time_budget_exhausted = (
                    budget_spec.time_budget_seconds is not None
                    and self.get_elapsed_seconds()
                    >= budget_spec.time_budget_seconds
                )
                cost_budget_exhausted = (
                    budget_spec.cost_budget_usd is not None
                    and self.get_cumulative_cost()
                    >= budget_spec.cost_budget_usd
                )
                budget_exhausted = (
                    time_budget_exhausted or cost_budget_exhausted
                )
                if budget_exhausted:
                    stop_detail = (
                        "time_budget"
                        if time_budget_exhausted
                        else "cost_budget"
                    )
                self._save_run_checkpoint(
                    status="completed" if node.should_stop else "running",
                    last_stop=stop_detail if budget_exhausted else None,
                )

                # Check if search strategy says stop
                if node.should_stop:
                    print("[Orchestrator] Stopping: goal achieved")
                    stopped_reason = "goal_achieved"
                    break

                if reserve_run_pending:
                    # The escrowed measurement is kapso-owned: whatever the
                    # agent self-reported, the authoritative FULL score of
                    # the reserve artifact comes from a frame run (the live
                    # reserve run did real 0.9-class work whose self-report
                    # died with a killed feedback call, landing score=None).
                    if (
                        node is not None
                        and self.search_strategy.registered_evaluator_id
                    ):
                        # Floored at the timing estimate: an endgame the
                        # build overran still grants the measurement its
                        # estimated slice — delivery-critical work, the
                        # same license the transition bridge has.
                        measured = self.search_strategy.run_bridge_evaluation(
                            node,
                            fidelity="full",
                            fraction=1.0,
                            deadline_seconds=(
                                max(
                                    self.budget_spec.time_budget_seconds
                                    - self.get_elapsed_seconds(),
                                    self.evaluation_maintainer.timing(
                                        1.0
                                    ).upper_seconds,
                                )
                                if self.budget_spec.time_budget_seconds
                                is not None
                                else None
                            ),
                        )
                        print(
                            "[Orchestrator] Reserve measurement "
                            f"{'recorded' if measured else 'failed'} for "
                            f"node {node.node_id}"
                        )
                    print(
                        "[Orchestrator] Reserve run complete — stopping "
                        "with the finalization window honored"
                    )
                    stopped_reason = "budget_exhausted"
                    stop_detail = "finalization_reserve"
                    self._save_run_checkpoint(
                        status="running",
                        last_stop=stop_detail,
                    )
                    break

                if budget_exhausted:
                    print("[Orchestrator] Stopping: budget exhausted")
                    stopped_reason = "budget_exhausted"
                    break

                print(
                    f"Experiment {i+1} completed with cumulative cost: ${self.get_cumulative_cost():.3f}", 
                    '#' * 100,
                    '\n', 
                    self.search_strategy.get_best_experiment(), 
                    '\n', 
                    '#' * 100
                )

            if stopped_reason == "max_iterations":
                # Persist even a zero-iteration slice so it can be resumed.
                self._save_run_checkpoint(status="running")
        finally:
            # Best-effort cleanup: prevents leaked sockets from KG/Episodic clients.
            
            # Close experiment history store
            if self.experiment_store:
                try:
                    self.experiment_store.close()
                except Exception:
                    pass
            
            # Close knowledge search only if the orchestrator created it.
            if getattr(self, "_owns_knowledge_search", False) and hasattr(self.knowledge_search, "close"):
                try:
                    self.knowledge_search.close()
                except Exception:
                    pass

        return SolveResult(
            best_experiment=self.search_strategy.get_deliverable_experiment(),
            final_feedback=self.last_feedback_result,
            stopped_reason=stopped_reason,
            iterations_run=iterations_run,
            total_cost=self.get_cumulative_cost(),
            cumulative_iterations=self.completed_iterations,
            stop_detail=stop_detail,
        )
