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
import subprocess
import sys
import threading
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

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
from kapso.execution.search_strategies.generic.implementation import render_inbox_section
from kapso.environment.handlers.base import ProblemHandler
from kapso.execution.inbox import (
    idea_line,
    continue_answered,
    load_requests,
    open_requests,
    render_pause_block,
    request_payload,
)
from kapso.core.prompt_loader import load_prompt, optional_block, render_prompt
from kapso.gated_mcp.presets import get_mcp_config, knowledge_tools_block
from kapso.harness.load import HarnessError
from kapso.harness.staging import PREPARED_DIR, TASK_STATEMENT_FILE, engine_session_record
from kapso.harness.task_statement import prepared_names, setup_statement_path, write_task_statement
from kapso.core.agent_manifest import load_agent_manifest
from kapso.core.cli_inference import CliInference
from kapso.core.config import load_config
from kapso.core.roles import BUILD_CONFIG_BLOCKS, SPEC_KEYS, check_maintainer, resolve_role, resolve_spec, validate_config
from kapso.execution.search_strategies.base import ExperimentResult, SearchNode
from kapso.execution.memories.experiment_memory import ExperimentHistoryStore
from kapso.execution.cost_estimator import (
    campaign_cost_used,
    load_cost_estimator,
    normalize_cost_config,
    normalize_cost_estimate,
    platform_cost_clause,
    selection_score,
)
from kapso.execution.iteration_evaluator import (
    IterationEvaluationContext,
    IterationEvaluationError,
    IterationEvaluator,
    normalize_failure_policy,
    normalize_result,
)
from kapso.execution.evaluation_integrity import (
    build_evaluation_manifest,
    launch_evaluation_manifest,
    manifest_fingerprint,
    verify_data_manifest,
)
from kapso.execution.observability import BuildStatus, knowledge_summary
from kapso.execution.run_checkpoint import (
    RunCheckpoint,
    RunCheckpointError,
    RunCheckpointCorruptError,
    RunCheckpointIncompatibleError,
    RunCheckpointStore,
    config_fingerprint,
)
from kapso.execution.budget import (
    ATTEMPT_WOULD_NOT_FIT,
    DELIVERY_SECONDS,
    FINISHING_CUT,
    BudgetLedger,
    BudgetSnapshot,
    BudgetSpec,
    CostEntry,
    admission_refusal,
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



# The harness setup session (kapso.harness): its inbox identity, the node
# it files requests under (none: it runs before the first experiment), its
# prompt, the line that offers the inbox, and the campaign directories it
# must leave alone.
HARNESS_SETUP_SESSION = "harness_setup"
HARNESS_SETUP_NODE = -1
HARNESS_SETUP_PROMPT = "execution/search_strategies/generic/prompts/harness_setup.md"
HARNESS_SETUP_INBOX_LINE = (
    "\n- If an input the task statement names is missing (a file, a variable), ask for it "
    "with the request_from_user tool, naming it exactly, and finish; the campaign pauses "
    "and your session runs again once the person has answered.\n"
)
PINNED_DIRECTORIES = ("kapso_evaluation", "kapso_datasets")


def _directory_manifest(path: Path) -> Dict[str, str]:
    """The files under a campaign directory by hash; {} when it is absent."""
    return build_evaluation_manifest(str(path)) if path.is_dir() else {}


@dataclass
class SolveResult:
    """Result from orchestrator.solve()."""
    best_experiment: Optional[ExperimentResult]
    final_feedback: Optional[FeedbackResult]
    stopped_reason: str  # "goal_achieved", "max_iterations", "budget_exhausted", "legacy_stop"
    iterations_run: int
    total_cost: float
    cumulative_iterations: int = 0
    # Which budget dial triggered a budget_exhausted stop: "time_budget",
    # "cost_budget", "attempt_would_not_fit" (admission), "finishing_cut"
    # (a finishing session not launched or cut), or None.
    stop_detail: Optional[str] = None
    # The open inbox requests when stopped_reason is "waiting_for_user":
    # what the person must answer before the campaign continues.
    requests: List[Dict[str, Any]] = field(default_factory=list)
    # The campaign's platform spend {used, budget, unit, refused} in its
    # own unit; None when cost-aware selection is disabled.
    platform_budget: Optional[Dict[str, Any]] = None


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
        config_path: The override file over the packaged config (None: the
            packaged defaults alone)
        coding_agent: The host every session runs on unless its role names
            one (`search_strategy.params.host`)
        goal: The goal text of the build (the rendered brief)
        iteration_evaluator: Optional observational callback for each finalized
            candidate Git ref
        iteration_evaluator_failure_policy: ``record`` or ``raise``
    """
    
    def __init__(
        self, 
        problem_handler: ProblemHandler,
        config_path: Optional[str] = None,
        coding_agent: Optional[str] = None,
        knowledge_search: Optional[KnowledgeSearch] = None,
        workspace_dir: Optional[str] = None,
        resume: bool = False,
        iteration_evaluator: Optional[IterationEvaluator] = None,
        iteration_evaluator_failure_policy: str = "record",
        initial_repo: Optional[str] = None,
        eval_dir: Optional[str] = None,
        data_dir: Optional[str] = None,
        goal: Optional[str] = None,
        strategy_params_overrides: Optional[Dict[str, Any]] = None,
        harness: Optional[Any] = None,
        experience: Optional[List[Dict[str, Any]]] = None,
    ):
        self.problem_handler = problem_handler
        # The loaded harness (kapso.harness.Harness) or None: threaded to the
        # strategy and every session, written into the checkpoint and the
        # status file, never part of the config fingerprint (its hash is
        # its own check, with its own message, in the facade).
        self.harness = harness
        # What the campaign is served (the experience package's records):
        # the loaded sources as plain records, written into the status
        # file and the checkpoint beside the harness; the facade's launch
        # record fingerprints the list.
        self.experience: List[Dict[str, Any]] = [dict(source) for source in experience or []]
        self._harness_setup: Optional[Dict[str, Any]] = None
        # The requests the harness setup session filed with the inbox: the
        # campaign pauses on them before iteration 1.
        self._setup_requests: List[Any] = []
        self._task_budget: Dict[str, Any] = {}
        # The problem's scoring direction: the history store ranks by it and
        # the checkpoint records it for the MCP history process.
        self.maximize_scoring = getattr(problem_handler, "maximize_scoring", True)
        self.config_path = config_path
        self.goal = goal or ""
        # Per-campaign strategy-param overrides merged over the mode
        # config's params (the facade's serving hook uses this to thread
        # bank_serving — same semantics as a benchmark runtime config,
        # without the temp file). Merged BEFORE fingerprinting, so resume
        # validation sees exactly what the strategy will run with.
        self._strategy_params_overrides = dict(strategy_params_overrides or {})
        # Load once before constructing shared services so model roles and retry
        # behavior apply consistently across strategy, memory, and commit calls.
        self.config = load_config(config_path)
        # The checks the walk cannot make (the hosts against the adapters,
        # the roles, the maintainer), here as in the facade and the doctor:
        # a benchmark runner constructs the campaign directly.
        validate_config(self.config, load_agent_manifest()["agents"])
        model_routes = self.config["models"]
        retry_config = self.config["retry"]
        # CLI-only inference (cli-only-inference-design.md): the object
        # every consumer receives as `llm` runs completions as coding-agent
        # sessions; embeddings and the cost meter delegate to the inner
        # LLMBackend unchanged.
        self.llm = CliInference(
            models=model_routes,
            retry_policy=retry_config,
            config_path=config_path,
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
        ) = self._resolve_search_strategy_config(coding_agent)
        self.cost_config = normalize_cost_config(self.strategy_params.get("cost"))
        self.strategy_params["cost"] = self.cost_config
        if self.cost_config["enabled"] and self.strategy_type != "generic":
            raise ValueError("cost.enabled requires the generic search strategy")
        self._platform_budget_exhausted = False
        # The campaign inbox (docs/research/evolve-hub-design.md v4) rides
        # the mode config into the strategy params — fingerprinted like
        # every other setting, so a resume never flips it silently.
        inbox_block = self.config["inbox"]
        if inbox_block is not None:
            self.strategy_params = {
                **self.strategy_params, "inbox": dict(inbox_block),
            }
        # Budgets are operator dials, not campaign identity — like
        # max_iterations (a solve() argument that was never fingerprinted).
        # Excluding the budget block keeps "resume with a bigger budget"
        # possible under strict resume validation.
        self._config_budget = dict(self.config["budget"])
        fingerprint_config = {
            "strategy_type": self.strategy_type,
            "strategy_params": self.strategy_params,
            "config": {key: self.config[key] for key in BUILD_CONFIG_BLOCKS},
        }
        self._provided_evaluation_manifest: Optional[Dict[str, str]] = None
        if eval_dir:
            self._provided_evaluation_manifest = launch_evaluation_manifest(
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
        self._checkpoint_lock = threading.Lock()
        self._heartbeat_stop = threading.Event()
        # Observability (design §2): the strategy constructor opens the
        # status file at workspace setup; solve() adopts it below once the
        # budget is resolved. Lives at <workspace>/.kapso/status.json.
        self.operation_status: Optional[BuildStatus] = None
        self._status_budget_total_min: Optional[float] = None
        self._status_reserve_seconds: Optional[float] = None
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
                cost_config=self.cost_config,
            )
            self._resume_checkpoint = checkpoint
            self._harness_setup = checkpoint.harness_setup
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
        
        self.cost_estimator = load_cost_estimator(self.cost_config)

        # Create experiment history store
        # Path is determined after search strategy creates workspace
        self.experiment_store: Optional[ExperimentHistoryStore] = None
        
        # Create feedback generator FIRST (needed by search strategy)
        self.feedback_generator = self._create_feedback_generator()
        
        # Track feedback for next iteration
        self.current_feedback: Optional[str] = (
            self._resume_checkpoint.current_feedback
            if self._resume_checkpoint is not None
            else None
        )
        self.last_feedback_result: Optional[FeedbackResult] = None
        
        # Create search strategy (uses feedback_generator)
        self.search_strategy = self._create_search_strategy(
            workspace_dir=workspace_dir,
            start_from_checkpoint=self.resume,
        )
        self.search_strategy.before_candidate_evaluation = self._admit_candidate

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
        self.fidelity_spec = FidelitySpec.resolve(self._config_budget["fidelity"])
        if self.cost_config["enabled"] and self.fidelity_spec.mode != "off":
            raise ValueError("cost.enabled requires budget.fidelity.mode: off")
        (
            self.evaluation_maintainer,
            self._max_change_requests,
            self._transition_freeze_fraction,
        ) = self._create_evaluation_maintainer()
        self._change_requests_filed = 0
        # The open requests a waiting_for_user stop leaves behind, as the
        # status file, the result and the on_status hook carry them.
        self._waiting_payload: List[Dict[str, Any]] = []
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
        # With no embedding model configured (`models.embedding` null, the
        # shipped default) the store is recency-only — a documented
        # capability absence, not an embedding call that fails at the
        # first node's bookkeeping.
        embedding_model = self.config["models"]["embedding"]
        self.experiment_store = ExperimentHistoryStore(
            json_path=experiment_history_path,
            goal=self.goal,
            llm=self.llm if embedding_model else None,
            cost_config=self.cost_config,
            maximize_scoring=self.maximize_scoring,
        )
        
        # Create knowledge search backend (or use provided instance).
        # This allows Kapso.build() to inject a concrete backend (e.g., kg_graph_search)
        # without relying on config defaults (which may point to a different backend).
        # The search over a loaded index comes from the facade, connected
        # from the index's record; nothing loaded is the null search. The
        # orchestrator builds no backend of its own: a config's
        # `knowledge_search` block is the index's parameters, not a switch
        # (the WP2 finding: a build with nothing loaded died here for a
        # missing embedding key).
        self.knowledge_search = (
            knowledge_search if knowledge_search is not None else KnowledgeSearchFactory.create_null()
        )
        self._owns_knowledge_search = False
    
    def _create_feedback_generator(self) -> FeedbackGenerator:
        """The judge: the `judge` role over the hosts table (kapso.core.roles)."""
        role = resolve_role(self.config["hosts"], self.strategy_params, "judge")
        return FeedbackGenerator(
            coding_agent_config=CodingAgentFactory.session_config(
                role, {"streaming": True, "timeout": role.timeout},
            ),
        )

    def _create_search_strategy(
        self,
        workspace_dir: Optional[str],
        start_from_checkpoint: bool,
    ) -> SearchStrategy:
        """The search strategy from the resolved config; the workspace's
        sessions carry the implementation role's host."""
        role = resolve_role(self.config["hosts"], self.strategy_params, "implementation")
        coding_agent_config = CodingAgentFactory.session_config(role, {})
        return SearchStrategyFactory.create(
            strategy_type=self.strategy_type,
            problem_handler=self.problem_handler,
            llm=self.llm,
            coding_agent_config=coding_agent_config,
            hosts=self.config["hosts"],
            params=self.strategy_params,
            workspace_dir=workspace_dir,
            start_from_checkpoint=start_from_checkpoint,
            initial_repo=self.initial_repo,
            eval_dir=self.eval_dir,
            evaluation_manifest=self._provided_evaluation_manifest,
            data_dir=self.data_dir,
            feedback_generator=self.feedback_generator,
            goal=self.goal,
            status_heartbeat_seconds=self._status_heartbeat_seconds(),
            harness=self.harness,
            experience=self.experience,
        )

    def _status_heartbeat_seconds(self) -> Optional[float]:
        """The status file's cadence: the same budget key that paces the
        checkpoint heartbeat (Rule 1 — one knob for both beats)."""
        interval = self._config_budget["checkpoint_heartbeat_seconds"]
        return float(interval) if interval else None

    def _resolve_search_strategy_config(self, coding_agent: Optional[str]) -> Tuple[str, Dict[str, Any]]:
        """The strategy's type and parameters: the config's block, the
        engine's own parameters over it, and `--coding-agent` as the host
        every role without a `cli` of its own runs on."""
        search_config = self.config["search_strategy"]
        params = {**search_config["params"], **self._strategy_params_overrides}
        if coding_agent:
            params["host"] = coding_agent
        return search_config["type"], params

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
        block = self.config["evaluation_maintainer"]
        check_maintainer(block)
        if not block["enabled"]:
            return None, 0, 1.0
        role = resolve_spec(
            self.config["hosts"], self.strategy_params["host"],
            {key: block[key] for key in SPEC_KEYS if key in block},
            name="evaluation_maintainer", origin="evaluation_maintainer",
        )
        agent_config = CodingAgentFactory.session_config(role, {"timeout": role.timeout})
        maintainer = EvaluationMaintainer(
            coding_agent_config=agent_config,
            workspace_dir=self.search_strategy.workspace_dir,
            # Single-sourced in the fidelity block: the fraction the policy
            # requests is the fraction the maintainer calibrates and registers.
            fast_fraction=self.fidelity_spec.eval_fast_fraction,
            subsample_seed=block["subsample_seed"],
            calibration_fraction=block["calibration_fraction"],
            calibration_timeout_seconds=block["calibration_timeout_seconds"],
            fast_variant_threshold_seconds=block["fast_variant_threshold_minutes"] * 60,
            overhead_factor=block["overhead_factor"],
            protected_data_paths=list(block["protected_data_paths"]),
        )
        freeze_fraction = block["transition_freeze_fraction"]
        if not 0.0 < freeze_fraction <= 1.0:
            raise ValueError(
                "transition_freeze_fraction must be in (0, 1], got "
                f"{freeze_fraction!r}"
            )
        return maintainer, int(block["max_change_requests"]), freeze_fraction

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
        self.search_strategy.record_eval_duration = (
            self.evaluation_maintainer.record_run
        )

    def _write_task_statement(self, **budget: Any) -> None:
        """`<campaign>/kapso_task.md`, rendered by the engine from the brief,
        the effective budget, the references, the setup session's statement
        and the prepared resources; nothing without a harness."""
        if self.harness is None:
            return
        campaign = Path(self.search_strategy.workspace_dir)
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", TASK_STATEMENT_FILE],
            cwd=campaign, capture_output=True, text=True,
        )
        if tracked.returncode == 0:
            raise HarnessError(
                f"the repository tracks {TASK_STATEMENT_FILE}, the task statement the engine writes at "
                "the campaign root for every session; rename or untrack it before building with a harness"
            )
        write_task_statement(
            campaign,
            brief_text=self.goal,
            harness=self.harness,
            budget=budget,
            evaluation_dir=self.eval_dir,
            data_dir=self.data_dir,
        )

    def _run_harness_setup(self) -> None:
        """The harness's setup session, once per fresh campaign, at the
        campaign root, when the harness has setup content: it writes its
        domain statement and prepared resources; the engine composes the
        task statement from them, verifies the pinned evaluation and data,
        records the step in the checkpoint and the status file, and pauses
        the campaign when the session asked the person for an input."""
        harness = self.harness
        if harness is None or not harness.has_content("setup") or self._harness_setup is not None:
            return
        strategy = self.search_strategy
        # The launch predicate, as for every session: no room under the
        # floor, no launch — and the setup is the campaign's precondition,
        # so the campaign stops here with the reason.
        room = strategy._room()
        if room is not None and room < strategy.budget_snapshot.min_agent_timeout_seconds:
            raise HarnessError(
                f"the budget leaves {max(room, 0.0) / 60:.1f} min for the harness setup session of "
                f"{harness.label()}, under the {strategy.budget_snapshot.min_agent_timeout_seconds:.0f} s floor "
                f"({strategy.budget_snapshot.finalization_reserve_seconds / 60:.1f} min of the budget are kept for "
                "finishing); give the campaign more minutes"
            )
        campaign = Path(strategy.workspace_dir)
        strategy._status_phase("harness_setup")
        host = strategy.implementation_role.cli
        staged = strategy.stage_harness_for("setup", str(campaign), HARNESS_SETUP_SESSION, host)
        pinned = {name: _directory_manifest(campaign / name) for name in PINNED_DIRECTORIES}

        inbox = getattr(strategy, "inbox_settings", None)
        inbox_on = bool(inbox and inbox.get("enabled"))
        mcp_servers: Dict[str, Any] = {}
        mcp_tools: List[str] = []
        inbox_agent_settings: Dict[str, Any] = {}
        if inbox_on:
            # The inbox id doubles as the host's session id, which Claude
            # Code requires to be a UUID; the setup session is found again
            # by its node id, not by this value.
            inbox_session_id = str(uuid.uuid4())
            mcp_servers, mcp_tools = get_mcp_config(
                gates=["inbox"],
                include_base_tools=False,
                gate_failure_policy=strategy.gate_failure_policy,
                inbox={
                    "KAPSO_INBOX_PATH": str(inbox["path"]),
                    "KAPSO_SESSION_ID": inbox_session_id,
                    "KAPSO_NODE_ID": str(HARNESS_SETUP_NODE),
                },
            )
            inbox_agent_settings = {
                "session_id": inbox_session_id,
                "inbox_path": str(inbox["path"]),
                "inbox_stop_grace_seconds": float(inbox["stop_grace_seconds"]),
            }
        allowed = [
            "Read", "Write", "Edit", "Bash",
            *[tool for tool in mcp_tools if tool.startswith("mcp__")],
            *staged.allowed_tools,
        ]
        config = CodingAgentFactory.session_config(strategy.implementation_role, {
            "env_strip": strategy.env_strip,
            "env_defaults": strategy.env_defaults,
            "mcp_servers": mcp_servers,
            "allowed_tools": allowed,
            "timeout": strategy._clamped_timeout(float(harness.settings["setup_minutes"]) * 60.0),
            "streaming": True,
            **inbox_agent_settings,
        }, staged)
        answered = (
            render_inbox_section(load_requests(inbox["path"]), dotenv_path=inbox.get("dotenv_path", ""))
            if inbox_on else ""
        )
        if inbox_on:
            # A restarted setup session carries the person's replies: the
            # setup's answered requests are continued by it, and the record
            # says so — a search node's session records its own on resume,
            # and the setup's were left `answered`, the request's promise
            # of a resumption unkept in the record (the S2 gate, 2026-10-30).
            continue_answered(inbox["path"], node=HARNESS_SETUP_NODE, session=inbox_session_id)
        prompt = render_prompt(load_prompt(HARNESS_SETUP_PROMPT), {
            "task_statement_path": str(campaign / TASK_STATEMENT_FILE),
            "statement_path": setup_statement_path(campaign).relative_to(campaign).as_posix(),
            "prepared_dir": PREPARED_DIR,
            "inbox_line": HARNESS_SETUP_INBOX_LINE if inbox_on else "",
            "knowledge_tools": (
                knowledge_tools_block("implementation", allowed) if inbox_on
                else "The harness's own servers and skills, listed below; no engine knowledge tools in this session."
            ),
            "domain_guidance": optional_block(
                staged.guidance + (("\n\n" + answered) if answered else "")
            ),
        })

        print(f"[Orchestrator] Harness setup session starting ({host}, {harness.label()})")
        started = time.monotonic()
        agent = CodingAgentFactory.create(config)
        agent.initialize(str(campaign))
        result = agent.generate_code(prompt)
        cost = agent.get_cumulative_cost()
        agent.cleanup()
        duration = time.monotonic() - started
        self.budget_ledger.record(
            CostEntry(component="harness_setup", cost_usd=float(cost), duration_seconds=duration)
        )
        for name, before in pinned.items():
            if _directory_manifest(campaign / name) != before:
                raise HarnessError(
                    f"the harness setup session of {harness.label()} changed {name}/, the campaign's "
                    f"pinned {'evaluation' if name == 'kapso_evaluation' else 'data'}; a harness prepares "
                    f"resources under {PREPARED_DIR}/ and leaves these alone"
                )
        if inbox_on:
            asked = [
                request for request in open_requests(load_requests(inbox["path"]))
                if request.node == HARNESS_SETUP_NODE
            ]
            if asked:
                self._setup_requests = asked
                print(
                    f"[Orchestrator] Harness setup asked the person ({len(asked)} request(s)); "
                    "the campaign pauses before its first experiment"
                )
                return
        if not result.success:
            raise HarnessError(f"the harness setup session of {harness.label()} failed: {result.error}")
        prepared = list(prepared_names(campaign))
        self._harness_setup = {
            "done": True,
            "prepared": prepared,
            "cost_usd": float(cost),
            "duration_seconds": round(duration, 1),
        }
        self._write_task_statement(**self._task_budget)
        status = getattr(strategy, "operation_status", None)
        if status is not None:
            status.update(harness_setup={"prepared": prepared})
        self._save_run_checkpoint(status="running")
        print(f"[Orchestrator] Harness setup done: prepared {', '.join(prepared) or 'nothing'}")

    def _stage_maintainer(self, module: str, label: str):
        """The harness's module for one maintainer session at the campaign
        root; None without a harness."""
        if self.harness is None:
            return None
        return self.search_strategy.stage_harness_for(
            module, self.search_strategy.workspace_dir, label,
            self.evaluation_maintainer.coding_agent_config.agent_type,
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
                staged=self._stage_maintainer("setup", "evaluation_setup"),
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
        if self.cost_config["enabled"]:
            raise ValueError(
                "cost.enabled does not support evaluator transitions that "
                "repeat a candidate's platform evaluation"
            )
        strategy = self.search_strategy
        new_evaluator_id = strategy.registered_evaluator_id
        old_evaluator_id = strategy.scores_evaluator_id
        fidelity, fraction = self._canonical_evaluation_params()
        # Affordability window, not the timing estimate: the bridge is
        # delivery-critical and may legitimately run inside the reserve —
        # but the window is never unbounded, and even past the cap one
        # calibrated full-eval upper is allowed (an unbounded window let a
        # deadlocked evaluator hold a campaign 6h, 2026-08-12).
        bridge_upper = self.evaluation_maintainer.timing(1.0).upper_seconds
        deadline = (
            max(
                self.budget_spec.time_budget_seconds
                - self.get_elapsed_seconds(),
                bridge_upper,
            )
            if self.budget_spec.time_budget_seconds is not None
            else bridge_upper
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
                and not node.cost_refusal
            ),
            key=lambda node: selection_score(node, {
                "cost": self.cost_config,
                "maximize_scoring": (
                    self.maximize_scoring if self.cost_config["enabled"] else True
                ),
            }),
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
            if self.cost_config["enabled"]:
                raise ValueError(
                    "cost.enabled does not support evaluation change requests "
                    "that repeat candidate evaluations"
                )
            # Late-transition freeze: an evaluator change accepted near the
            # deadline orphans final selection (only the bridge run is
            # head-stamped — observed 2026-08-16, one eligible final from a
            # 12h ladder). Past the freeze fraction of a time budget, defer
            # the request instead of filing it.
            if (
                self.budget_spec.time_budget_seconds is not None
                and self.get_elapsed_seconds()
                >= self._transition_freeze_fraction
                * self.budget_spec.time_budget_seconds
            ):
                print(
                    "[Orchestrator] Evaluation change request deferred: "
                    f"budget fraction elapsed exceeds the transition freeze "
                    f"({self._transition_freeze_fraction:.2f}); the ruler "
                    "stays fixed through final selection"
                )
                return
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
                ),
                staged=self._stage_maintainer("verification", f"evaluation/change_{candidate.node_id}"),
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

    def _goal_already_achieved(self) -> bool:
        """Whether the campaign already stopped on its goal: the latest
        node voted stop and the handler honours the vote. True only on a
        resume whose delivery never completed (the checkpoint stays
        `running` until it does)."""
        history = self.search_strategy.get_experiment_history()
        return bool(history) and bool(history[-1].should_stop) and bool(self.strategy_params["honor_agent_stop"])

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
            cost_config=self.cost_config,
            maximize_scoring=self.maximize_scoring,
            harness=self.harness.ref_dict() if self.harness is not None else None,
            harness_setup=self._harness_setup,
            experience=self.experience,
        )
        with self._checkpoint_lock:
            self.checkpoint_store.save(checkpoint)

    def _status_budget_view(self) -> Dict[str, Any]:
        """The status file's time-only budget payload (observability §1):
        the clock, the budget and the round's reserve for finishing."""
        return {
            "elapsed_min": round(self.get_elapsed_seconds() / 60, 1),
            "total_min": self._status_budget_total_min,
            "reserve_min": (
                round(self._status_reserve_seconds / 60, 1)
                if self._status_reserve_seconds is not None and self._status_budget_total_min is not None
                else None
            ),
        }

    def _delivery_seconds(self) -> float:
        """The delivery's protected slice: the fidelity measurement when a
        maintainer measures the delivered artifact, the final checkout's
        seconds otherwise."""
        if self._fidelity_active:
            return self.evaluation_maintainer.timing(1.0).upper_seconds
        return DELIVERY_SECONDS

    def _round_reserve_seconds(self, budget_spec: BudgetSpec) -> float:
        """The reserve this round keeps out of the search: the set one,
        else derived from the obligations — the harness's verification
        minutes when it verifies, the judge's last observed duration (the
        config's judge reserve before one: a judge takes minutes, and a
        reserve at the floor let the first attempt's verification run the
        remainder down to 59 s for it, under the floor, the hour's attempt
        then unjudged — the release gate, 2026-10-30), the delivery's
        slice; either bounded by a quarter of the budget."""
        bounded = budget_spec.bounded_reserve_seconds()
        if bounded is not None:
            return bounded
        verification = (
            float(self.harness.settings["verification_minutes"]) * 60.0
            if self.harness is not None and self.harness.has_content("verification")
            else 0.0
        )
        return budget_spec.bound_reserve(verification + self._judge_reserve_seconds(budget_spec) + self._delivery_seconds())

    def _judge_reserve_seconds(self, budget_spec: BudgetSpec) -> float:
        """What this round keeps for the judge: the last judge's observed
        duration in the campaign, the config's reserve before one. The
        round's reserve holds it back from the search, and the strategy's
        finishing stages leave it standing (the snapshot carries it)."""
        observed = [
            node.phase_telemetry["feedback"]["duration_seconds"]
            for node in self.search_strategy.get_experiment_history()
            if "feedback" in node.phase_telemetry
        ]
        return observed[-1] if observed else budget_spec.judge_reserve_seconds

    def _platform_budget_view(self) -> Optional[Dict[str, Any]]:
        """The campaign's platform spend against its budget, in the
        campaign's own unit; None when cost is disabled."""
        if not self.cost_config["enabled"]:
            return None
        history = self.search_strategy.get_experiment_history()
        return {
            "used": campaign_cost_used(history, self.cost_config),
            "budget": self.cost_config["campaign_budget"],
            "unit": self.cost_config["unit"],
            "refused": sum(1 for node in history if node.cost_refusal),
        }

    def _status_platform_budget_payload(self) -> Dict[str, Any]:
        """The status file's platform_budget key, present only when cost is enabled."""
        view = self._platform_budget_view()
        return {} if view is None else {"platform_budget": view}

    def _run_checkpoint_heartbeat(self, interval_seconds: float) -> None:
        """Daemon loop: keep the durable clock fresh between boundary
        checkpoints so a preemption cannot rewind the budget on resume.
        The same beat refreshes the observability status file — a build's
        liveness signal rides this existing daemon (observability §2)."""
        while not self._heartbeat_stop.wait(interval_seconds):
            if self.operation_status is not None:
                self.operation_status.heartbeat(
                    budget=self._status_budget_view()
                )
            if self.checkpoint_store is None:
                continue
            with self._checkpoint_lock:
                self.checkpoint_store.heartbeat_elapsed(
                    self.get_elapsed_seconds()
                )

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

    def _waiting_requests(self) -> List[Any]:
        """Open inbox requests of nodes waiting for the person; empty for
        strategies without an inbox."""
        if self._setup_requests:
            return list(self._setup_requests)
        hook = getattr(self.search_strategy, "waiting_requests", None)
        return list(hook()) if callable(hook) else []

    def _pause_for_user(self, requests: List[Any]) -> None:
        """The campaign pauses: print what the person must answer, save a
        resumable checkpoint marked waiting_for_user, and remember the
        requests for the status file and the result (design v4 §4.3)."""
        ideas = {
            node.node_id: idea_line(node.solution)
            for node in self.search_strategy.get_experiment_history()
        }
        campaign_dir = str(Path(self.search_strategy.workspace.workspace_dir).resolve())
        print("\n" + render_pause_block(requests, ideas, campaign_dir))
        self._waiting_payload = [
            request_payload(request, ideas.get(request.node, ""))
            for request in requests
        ]
        self._save_run_checkpoint(status="running", last_stop="waiting_for_user")
        self.operation_status.note(
            "waiting on the person: requests "
            + ", ".join(f"#{request.id}" for request in requests),
            requests=list(self._waiting_payload),
        )

    def _admit_candidate(self, candidate: SearchNode) -> bool:
        """Estimate a finalized candidate before any platform evaluation."""
        if not self.cost_config["enabled"]:
            return True
        if not candidate.branch_name:
            raise ValueError("Cost estimation requires a finalized candidate branch")
        with self.search_strategy.workspace.materialize_ref(
            candidate.branch_name
        ) as materialized_dir:
            snapshot = SearchNode.from_dict(deepcopy(candidate.to_dict()))
            snapshot.workspace_dir = str(materialized_dir)
            estimate = normalize_cost_estimate(
                self.cost_estimator(IterationEvaluationContext(
                    iteration=self.completed_iterations + 1,
                    goal=self.goal,
                    workspace_dir=Path(materialized_dir),
                    git_ref=candidate.branch_name,
                    parent_ref=candidate.parent_branch_name or "main",
                    node=snapshot,
                )),
                self.cost_config["unit"],
            )
        candidate.estimated_cost = estimate
        used = campaign_cost_used(
            [node for node in self.search_strategy.get_experiment_history()
             if node.node_id != candidate.node_id],
            self.cost_config,
        )
        cap = self.cost_config["cap_per_candidate"]
        campaign_budget = self.cost_config["campaign_budget"]
        # The spend before this candidate, the figure the refusal records.
        budget_text = (
            f"{used}/{float(campaign_budget)}" if campaign_budget is not None
            else f"{used}"
        )
        reason = None
        if self._platform_budget_exhausted or (
            campaign_budget is not None
            and used + estimate["amount"] > campaign_budget
        ):
            reason = "campaign_budget"
            cap = campaign_budget
            self._platform_budget_exhausted = True
        elif cap is not None and estimate["amount"] > cap:
            reason = "cap_per_candidate"
        if reason is None:
            print(
                f"[Orchestrator] node {candidate.node_id} estimated "
                f"{estimate['amount']} {estimate['unit']}; campaign "
                f"{budget_text} used; running evaluation"
            )
            return True
        print(
            f"[Orchestrator] node {candidate.node_id} refused: {reason} "
            f"(estimate {estimate['amount']} {estimate['unit']}, {budget_text} used)"
        )
        candidate.score = None
        candidate.evaluation_valid = False
        candidate.should_stop = False
        candidate.cost_refusal = {
            "estimate": estimate,
            "cap": cap,
            "reason": reason,
            "campaign_cost_used": used,
        }
        candidate.feedback = (
            f"Too expensive by estimate: {estimate['amount']} {estimate['unit']}; "
            f"{reason} cap {cap} {estimate['unit']}. "
            f"Campaign cost used: {used} {estimate['unit']}. "
            "Candidate was not evaluated. Propose a cheaper design."
        )
        return False

    def _platform_budget_spent(self) -> bool:
        """True once the campaign's recorded platform spend has reached its budget."""
        campaign_budget = self.cost_config["campaign_budget"]
        return (
            self.cost_config["enabled"]
            and campaign_budget is not None
            and campaign_cost_used(
                self.search_strategy.get_experiment_history(), self.cost_config
            ) >= campaign_budget
        )

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
            if candidate.cost_refusal or (
                self.cost_config["enabled"] and candidate.estimated_cost is None
            ):
                continue
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
        on_status: Optional[Callable[[Dict[str, Any]], None]] = None,
        finish: Optional[Callable[["SolveResult"], None]] = None,
    ) -> SolveResult:
        """
        Run the main experimentation loop. `finish`, when given, runs
        inside the protected region after the search and before the
        terminal publish: the caller's delivery (the checkout and the
        record), so a watcher that sees `done` finds the record, and a
        delivery that raises is published as the campaign's failure with
        a checkpoint a resume can finish from.
        
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
        # The protected region is the whole of solve(): from its first
        # statement on, an exception in flight is published as the
        # campaign's failure in the `finally` below (wp6-observability.md,
        # change 6), and the delivery the caller hands in as `finish` runs
        # inside it too.
        result: Optional[SolveResult] = None
        try:
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
            self._fidelity_active = False
            self.budget_ledger.start_clock()
            # With a harness, the task statement every session reads first is
            # written before any session runs (and again after harness setup).
            # The effective budget, as resolved: what a session reading the
            # statement must plan against.
            self._task_budget = dict(
                experiments=experiment_max_iter,
                minutes=budget_spec.time_budget_seconds / 60.0 if budget_spec.time_budget_seconds is not None else None,
                usd=budget_spec.cost_budget_usd,
                reserve_minutes=(
                    budget_spec.bounded_reserve_seconds() / 60.0
                    if budget_spec.bounded_reserve_seconds() is not None else None
                ),
            )
            self._write_task_statement(**self._task_budget)
            # The harness's own setup session, once, before the evaluation is
            # registered: a domain statement and prepared resources. The data
            # copy is pinned across it and across the maintainer's session,
            # which carries the harness's setup module too.
            campaign = Path(self.search_strategy.workspace_dir)
            pinned_data = _directory_manifest(campaign / "kapso_datasets") if self.harness is not None else None
            if self.harness is not None:
                # The setup session's clamp reads a budget view too: without one
                # a one-minute campaign would start a twenty-minute session.
                self.search_strategy.observe_budget(BudgetSnapshot(
                    iteration_index=0,
                    max_iterations=experiment_max_iter,
                    elapsed_seconds=self.get_elapsed_seconds(),
                    cost_usd=self.get_cumulative_cost(),
                    time_budget_seconds=budget_spec.time_budget_seconds,
                    cost_budget_usd=budget_spec.cost_budget_usd,
                    finalization_reserve_seconds=self._round_reserve_seconds(budget_spec),
                    min_agent_timeout_seconds=budget_spec.min_agent_timeout_seconds,
                    judge_reserve_seconds=self._judge_reserve_seconds(budget_spec),
                    delivery_seconds=self._delivery_seconds(),
                ))
            self._run_harness_setup()
            # The maintainer's setup transaction runs inside the budgeted clock,
            # before iteration 1; on resume it validates registry consistency.
            # Not while the setup session's question stands: the campaign
            # pauses before spending the maintainer's session on an input the
            # person has not given.
            if not self._setup_requests:
                self._ensure_evaluation_registered()
                if pinned_data is not None and _directory_manifest(campaign / "kapso_datasets") != pinned_data:
                    raise HarnessError(
                        "the evaluation maintainer's setup session changed kapso_datasets/, the campaign's "
                        "pinned data, with the harness's setup module staged; a session prepares resources "
                        f"under {PREPARED_DIR}/ and leaves the data alone"
                    )

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
                # Auto mode prices a full run from restored history: a fresh
                # campaign is conservatively tiered; every resume re-prices.
                fidelity_active = fidelity_policy.enabled(
                    budget_spec.time_budget_seconds,
                    self.search_strategy.get_experiment_history(),
                    self._probe_estimate_seconds(budget_spec),
                )
            self._fidelity_active = fidelity_active
            # Replay a pending transition or anchor restored scores on the
            # current registry head before any selection happens.
            self._reconcile_evaluator_state()
            reserve_run_pending = False

            stopped_reason = "max_iterations"  # default
            self._platform_budget_exhausted = False
            stop_detail: Optional[str] = None
            iterations_run = 0

            # Bootstrap checkpoint: registration and reconciliation are paid,
            # durable work — a crash at any later point must resume instead of
            # restarting the campaign (and re-buying the maintainer setup).
            self._save_run_checkpoint(status="running")
            heartbeat_seconds = self._status_heartbeat_seconds()

            # Observability (design §1/§2): the status file lives with the
            # campaign's artifacts. The strategy constructor opened it at
            # workspace setup (phase `setup`, state `starting`); the loop adopts
            # it here — the caller's hook, the budget view, and from now on the
            # checkpoint-heartbeat daemon below beats it too. A strategy built
            # outside the base constructor (a test double) gets one now.
            self._status_budget_total_min = (
                round(budget_spec.time_budget_seconds / 60, 1)
                if budget_spec.time_budget_seconds is not None else None
            )
            # A new slice on a finished status (solve() called again on the
            # same orchestrator) opens a fresh file, as every slice always did.
            status = getattr(self.search_strategy, "operation_status", None)
            if status is None or status.terminal:
                status = BuildStatus(
                    Path(self.search_strategy.workspace.workspace_dir)
                    / ".kapso" / "status.json",
                    heartbeat_seconds=heartbeat_seconds,
                    harness=self.harness.ref_dict() if self.harness is not None else None,
                    experience=self.experience,
                )
            status.observe(on_status)
            status.update(
                heartbeat_seconds=heartbeat_seconds,
                budget=self._status_budget_view(),
                **self._status_platform_budget_payload(),
            )
            self.operation_status = status
            self.search_strategy.operation_status = status
            print(f"status: {self.operation_status.path}")

            if heartbeat_seconds and self.checkpoint_store is not None:
                threading.Thread(
                    target=self._run_checkpoint_heartbeat,
                    args=(float(heartbeat_seconds),),
                    name="run-checkpoint-heartbeat",
                    daemon=True,
                ).start()

            # Get problem context once (experiment history is accessed via MCP)
            problem = self.problem_handler.get_problem_context()

            for i in range(experiment_max_iter):
                # A node waiting for the person means no new work: the
                # campaign pauses until the reply is in (design v4 §4.3).
                waiting = self._waiting_requests()
                if waiting:
                    self._pause_for_user(waiting)
                    stopped_reason = "waiting_for_user"
                    break
                if self._goal_already_achieved():
                    # A resume of a campaign that stopped on its goal but
                    # never delivered: nothing to search, the delivery is
                    # what is left.
                    stopped_reason = "goal_achieved"
                    break

                # The platform budget is read from history before the
                # iteration spends anything: a slice whose recorded spend has
                # reached campaign_budget stops here instead of ideating and
                # building a candidate that admission must then refuse. The
                # refusal flag resets with every solve(); the history does not.
                if self._platform_budget_spent():
                    print("[Orchestrator] Stopping: platform budget spent")
                    stopped_reason = "cost_budget_exhausted"
                    stop_detail = stopped_reason
                    self._save_run_checkpoint(
                        status="running", last_stop=stopped_reason,
                    )
                    break

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
                    else self._round_reserve_seconds(budget_spec)
                )
                self._status_reserve_seconds = effective_reserve_seconds
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
                    judge_reserve_seconds=self._judge_reserve_seconds(budget_spec),
                    delivery_seconds=self._delivery_seconds(),
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

                # Admission (the budget's end, wp4-config.md): an ordinary
                # round is admitted when the searchable remainder covers
                # the larger of the floor and the last completed round. With
                # fidelity active and no full-size result yet, a refusal
                # grants the guaranteed reserve run instead of stopping.
                remaining_after_reserve = snapshot.remaining_after_reserve
                needed_seconds = budget_spec.min_iteration_seconds
                last_round = self.search_strategy.last_round_seconds
                if last_round is not None:
                    needed_seconds = max(needed_seconds, last_round)
                # A continuation is not a round: a node suspended on a
                # request the person has answered resumes its own session
                # whatever the remainder, since the attempt is under way and
                # the person answered (the release gate's S5 saw a resume
                # refused on the paused attempt's own elapsed time — "the
                # last took 18.2 min" — and the reply never reach the
                # session). The continuation's result is judged like any
                # finished node; the next admission sees the real remainder.
                continuation = self.search_strategy.continues_an_answered_node()
                if continuation and remaining_after_reserve is not None and remaining_after_reserve < needed_seconds:
                    print(
                        "[Orchestrator] Continuing the node that waited on the inbox: "
                        f"the remaining {max(remaining_after_reserve, 0.0) / 60:.1f} min would not admit a new attempt, "
                        "and a continuation is not one"
                    )
                if not continuation and remaining_after_reserve is not None and remaining_after_reserve < needed_seconds:
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
                            judge_reserve_seconds=self._judge_reserve_seconds(budget_spec),
                            delivery_seconds=self._delivery_seconds(),
                        )
                        self.search_strategy.observe_budget(snapshot)
                    else:
                        prose = admission_refusal(
                            remaining_after_reserve, needed_seconds, last_round,
                        )
                        print(f"[Orchestrator] Stopping: {prose}")
                        self.operation_status.note(f"stopping: {prose}")
                        stopped_reason = "budget_exhausted"
                        stop_detail = ATTEMPT_WOULD_NOT_FIT
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
                self.operation_status.note(
                    f"iteration {iterations_run} started "
                    f"(budget {budget_progress:.0f}%)",
                    iteration=iterations_run,
                    budget=self._status_budget_view(),
                )

                # Build context with problem and feedback
                # Experiment history is accessed via MCP tools by the agent
                context = problem
                if self.current_feedback:
                    context = f"{problem}\n\n## Feedback from Previous Iteration\n\n{self.current_feedback}\n\nPlease address the above feedback in this iteration."
                
                # Run one iteration of search strategy
                # Search strategy handles: solution generation, implementation, feedback
                # Returns SearchNode with all data including should_stop
                # A node that asked the person is not finalized: it stays
                # out of the "seen" set so its continuation counts as new.
                previous_node_ids = {
                    candidate.node_id
                    for candidate in (
                        self.search_strategy.get_experiment_history()
                    )
                    if not getattr(candidate, "suspended", False)
                }
                node = self.search_strategy.run(
                    context, 
                    budget_progress=budget_progress
                )
                
                # Skip if no result (shouldn't happen but be safe)
                if node is None:
                    print(f"[Orchestrator] Warning: No result from iteration {i+1}")
                    continue

                finalized_candidates = [
                    candidate
                    for candidate in self._new_candidates(
                        previous_node_ids,
                        self.search_strategy.get_experiment_history(),
                        node,
                    )
                    if not getattr(candidate, "suspended", False)
                ]
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

                if not self._platform_budget_exhausted:
                    self._route_change_requests([
                        candidate for candidate in finalized_candidates
                        if not candidate.cost_refusal
                    ])
                
                # A sibling's campaign-budget refusal takes precedence
                # over the representative waiting for user input.
                if (
                    getattr(node, "suspended", False)
                    and not self._platform_budget_exhausted
                ):
                    # The session asked the person and was stopped: no
                    # judgement, no iteration counted — pause until the
                    # reply is in (design v4 §4.3).
                    self._pause_for_user(self._waiting_requests())
                    stopped_reason = "waiting_for_user"
                    break

                # Log result
                print(f"[Orchestrator] Iteration {i+1} result:")
                print(f"  - Score: {node.score}")
                print(f"  - Should stop: {node.should_stop}")
                print(f"  - Evaluation valid: {node.evaluation_valid}")
                print(f"  - Feedback: {node.feedback or ''}")
                knowledge = knowledge_summary(node.phase_telemetry)
                if knowledge:
                    print(f"  - Knowledge consulted: {knowledge}")
                cost_clause = platform_cost_clause(node, self.cost_config)
                if cost_clause:
                    print(f"  - Platform cost: {cost_clause}")

                best_node = self.search_strategy.get_best_experiment()
                external_status = {}
                if self.iteration_evaluator is not None:
                    external_status["external_evaluation"] = {
                        "node": node.node_id,
                        "metadata": node.external_evaluation_metadata,
                        "error": node.external_evaluation_error,
                    }
                self.operation_status.note(
                    f"node {node.node_id} completed score={node.score}"
                    + (f"; knowledge: {knowledge}" if knowledge else "")
                    + (f"; {cost_clause}" if cost_clause else ""),
                    last={"score": node.score, "node": node.node_id},
                    best=(
                        {
                            "score": getattr(best_node, "score", None),
                            "node": getattr(best_node, "node_id", None),
                        }
                        if best_node is not None
                        else None
                    ),
                    **external_status,
                    **self._status_platform_budget_payload(),
                )
                
                # Store feedback result for return value
                if node.feedback:
                    from kapso.execution.search_strategies.generic import FeedbackResult
                    self.last_feedback_result = FeedbackResult(
                        stop=node.should_stop,
                        evaluation_valid=node.evaluation_valid,
                        feedback=node.feedback,
                        score=node.score,
                        cards_load_bearing=list(node.cards_load_bearing),
                    )

                self.current_feedback = node.feedback
                self.completed_iterations += 1
                if self._platform_budget_exhausted:
                    stopped_reason = "cost_budget_exhausted"
                    stop_detail = stopped_reason
                    self._save_run_checkpoint(
                        status="running", last_stop=stopped_reason,
                    )
                    break
                cut = self.search_strategy.finishing_cut
                if cut is not None:
                    # A finishing session of this round was not launched
                    # or was cut: the budget's end, said by name.
                    prose = f"a finishing session was cut: {cut}"
                    print(f"[Orchestrator] Stopping: {prose}")
                    self.operation_status.note(f"stopping: {prose}")
                    stopped_reason = "budget_exhausted"
                    stop_detail = FINISHING_CUT
                    self._save_run_checkpoint(status="running", last_stop=stop_detail)
                    break
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
                # A stop vote only ends the campaign when the config says the
                # task has a goal to reach: open-ended maximization has no
                # "goal achieved", so there the vote is advisory and the
                # budget is the only clock.
                stopping = (
                    node.should_stop
                    and bool(self.strategy_params["honor_agent_stop"])
                )
                # `completed` is saved after the delivery, below.
                self._save_run_checkpoint(
                    status="running",
                    last_stop=stop_detail if budget_exhausted else None,
                )

                if stopping:
                    print("[Orchestrator] Stopping: goal achieved")
                    stopped_reason = "goal_achieved"
                    break
                if node.should_stop:
                    print("[Orchestrator] Agent voted stop — advisory only; "
                          "this handler runs to the time budget")

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
                                else self.evaluation_maintainer.timing(
                                    1.0
                                ).upper_seconds
                            ),
                        )
                        print(
                            "[Orchestrator] Reserve measurement "
                            f"{'recorded' if measured else 'failed'} for "
                            f"node {node.node_id}"
                        )
                    prose = "the escrowed full-size attempt was the last the budget covers"
                    print(f"[Orchestrator] Stopping: {prose}")
                    self.operation_status.note(f"stopping: {prose}")
                    stopped_reason = "budget_exhausted"
                    stop_detail = ATTEMPT_WOULD_NOT_FIT
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
            result = SolveResult(
                best_experiment=self.search_strategy.get_deliverable_experiment(),
                final_feedback=self.last_feedback_result,
                stopped_reason=stopped_reason,
                iterations_run=iterations_run,
                total_cost=self.get_cumulative_cost(),
                cumulative_iterations=self.completed_iterations,
                stop_detail=stop_detail,
                requests=(
                    list(self._waiting_payload)
                    if stopped_reason == "waiting_for_user" else []
                ),
                platform_budget=self._platform_budget_view(),
            )
            if finish is not None:
                finish(result)
            if stopped_reason == "goal_achieved":
                # Completed only once delivered: a delivery that raised
                # leaves a checkpoint a resume can finish from.
                self._save_run_checkpoint(status="completed")
        finally:
            self._heartbeat_stop.set()
            # Terminal status without an except: sys.exc_info() inside a
            # finally sees any in-flight exception, so the state machine
            # closes on both paths and the error still propagates
            # unhandled (Rule 2).
            in_flight = sys.exc_info()[1]
            # Before the loop adopted the status (an exception in the
            # harness setup), the strategy's own is the campaign's.
            status = self.operation_status or getattr(self.search_strategy, "operation_status", None)
            if status is None or status.terminal:
                pass
            elif in_flight is not None:
                status.failed(
                    in_flight, budget=self._status_budget_view()
                )
            else:
                status.done(
                    stopped_reason=stopped_reason,
                    stop_detail=stop_detail,
                    budget=self._status_budget_view(),
                    **self._status_platform_budget_payload(),
                    **(
                        {"requests": list(self._waiting_payload)}
                        if stopped_reason == "waiting_for_user" else {}
                    ),
                )
            # Best-effort cleanup: prevents leaked sockets from KG/Episodic clients.

            # Close knowledge search only if the orchestrator created it.
            if getattr(self, "_owns_knowledge_search", False) and hasattr(self.knowledge_search, "close"):
                try:
                    self.knowledge_search.close()
                except Exception:
                    pass

        return result
