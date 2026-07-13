# Search Strategy Base Classes
#
# Minimal base class for all search strategies.
# Contains only shared infrastructure and abstract method signatures.
#
# To create a new strategy:
# 1. Subclass SearchStrategy
# 2. Implement abstract methods: run(), get_experiment_history(), get_best_experiment(), etc.
# 3. Register with @register_strategy("your_name") decorator in factory.py

import os
import shutil
import time
import uuid
import logging
import math
from abc import ABC, abstractmethod
from dataclasses import MISSING, dataclass, field, fields
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from kapso.execution.types import ContextData
from kapso.execution.experiment_workspace.experiment_workspace import ExperimentWorkspace
from kapso.execution.coding_agents.base import CodingAgentConfig
from kapso.environment.handlers.base import ProblemHandler
from kapso.core.llm import LLMBackend
from kapso.execution.memories.repo_memory import RepoMemoryManager
from kapso.execution.fidelity import (
    FIDELITIES,
    EvaluationAttempt,
)
from kapso.execution.evaluation_integrity import (
    AGENT_GENERATED,
    PROVIDED,
    VALID_PROVENANCE,
    build_evaluation_manifest,
    manifest_fingerprint,
    verify_evaluation_tree,
)

# Avoid circular import - FeedbackGenerator is optional
if TYPE_CHECKING:
    from kapso.execution.budget import BudgetSnapshot
    from kapso.execution.fidelity import FidelityDecision
    from kapso.execution.search_strategies.generic import FeedbackGenerator


logger = logging.getLogger(__name__)


@dataclass
class SearchNode:
    """
    Unified node structure for search strategies.

    Accumulates data through the node lifecycle:
    1. Solution generation -> solution populated
    2. Implementation -> branch_name, code_changes_summary populated
    3. Evaluation -> evaluation_script_path, evaluation_output populated
    4. Feedback -> feedback, score, should_stop populated
    """
    node_id: int
    parent_node_id: Optional[int] = None
    
    # Step 1: Solution generation
    solution: str = ""
    
    # Step 2: Implementation
    branch_name: str = ""
    parent_branch_name: str = ""  # Parent branch for git diff reference
    code_changes_summary: str = ""
    agent_output: str = ""  # Raw output from developer agent
    
    # Step 3: Evaluation (extracted from agent output or result.json)
    evaluation_script_path: str = ""
    evaluation_output: str = ""
    
    # Step 4: Feedback
    feedback: str = ""
    score: Optional[float] = None
    should_stop: bool = False
    evaluation_valid: bool = True
    evaluation_provenance: str = AGENT_GENERATED
    evaluation_integrity_error: str = ""

    # Observational metrics from a caller-owned iteration evaluator. These do
    # not participate in search, stopping, or best-candidate selection.
    metrics: Dict[str, float] = field(default_factory=dict)
    primary_metric: Optional[str] = None
    external_evaluation_metadata: Dict[str, Any] = field(default_factory=dict)
    external_evaluation_error: str = ""

    # Per-iteration budget telemetry: what this experiment actually cost.
    # Absent values stay None/empty — unknowns are never zero-filled.
    duration_seconds: Optional[float] = None
    cost_usd: Optional[float] = None
    started_at: str = ""  # ISO-8601 UTC
    phase_telemetry: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # {"ideation": {"cost_usd": ..., "duration_seconds": ...}, "implementation": ..., "feedback": ...}

    # Fidelity: the workload profile this node ran at, its promotion lineage,
    # and its append-only versioned measurements (see execution/fidelity.py).
    build_fidelity: str = "full"
    eval_fidelity: str = "full"
    promoted_from: Optional[int] = None
    evaluation_attempts: List[EvaluationAttempt] = field(default_factory=list)
    
    # Metadata
    had_error: bool = False
    error_message: str = ""
    workspace_dir: str = ""
    code_diff: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize stable base-node fields to JSON-compatible data."""
        values = {}
        for item in fields(SearchNode):
            if hasattr(self, item.name):
                values[item.name] = getattr(self, item.name)
            elif item.default is not MISSING:
                values[item.name] = item.default
            elif item.default_factory is not MISSING:
                values[item.name] = item.default_factory()
        values["evaluation_attempts"] = [
            attempt.to_dict() for attempt in values["evaluation_attempts"]
        ]
        return values

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchNode":
        """Load a node while tolerating fields added by newer versions."""
        if not isinstance(data, dict):
            raise ValueError("Search node state must be an object")
        allowed = {item.name for item in fields(SearchNode)}
        values = {key: value for key, value in data.items() if key in allowed}
        if "node_id" not in values:
            raise ValueError("Search node state is missing node_id")
        node_id = values["node_id"]
        if (
            isinstance(node_id, bool)
            or not isinstance(node_id, int)
            or node_id < 0
        ):
            raise ValueError("Search node node_id must be a non-negative integer")
        parent_node_id = values.get("parent_node_id")
        if parent_node_id is not None and (
            isinstance(parent_node_id, bool)
            or not isinstance(parent_node_id, int)
            or parent_node_id < 0
        ):
            raise ValueError(
                "Search node parent_node_id must be null or non-negative"
            )

        string_fields = {
            "solution",
            "branch_name",
            "parent_branch_name",
            "code_changes_summary",
            "agent_output",
            "evaluation_script_path",
            "evaluation_output",
            "feedback",
            "error_message",
            "workspace_dir",
            "code_diff",
            "external_evaluation_error",
            "evaluation_integrity_error",
            "started_at",
        }
        invalid_strings = sorted(
            name
            for name in string_fields
            if name in values and not isinstance(values[name], str)
        )
        if invalid_strings:
            raise ValueError(
                "Search node string fields are invalid: "
                + ", ".join(invalid_strings)
            )

        for name in ("should_stop", "evaluation_valid", "had_error"):
            if name in values and not isinstance(values[name], bool):
                raise ValueError(f"Search node {name} must be a boolean")

        provenance = values.get("evaluation_provenance", AGENT_GENERATED)
        if (
            not isinstance(provenance, str)
            or provenance not in VALID_PROVENANCE
        ):
            raise ValueError(
                "Search node evaluation_provenance must be 'provided' or "
                "'agent_generated'"
            )

        score = values.get("score")
        if score is not None and (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            raise ValueError("Search node score must be finite or null")

        for name in ("duration_seconds", "cost_usd"):
            value = values.get(name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                raise ValueError(
                    f"Search node {name} must be finite and non-negative "
                    "or null"
                )

        phase_telemetry = values.get("phase_telemetry", {})
        if not isinstance(phase_telemetry, dict):
            raise ValueError("Search node phase_telemetry must be an object")
        for phase_name, phase_values in phase_telemetry.items():
            if not isinstance(phase_name, str) or not isinstance(
                phase_values, dict
            ):
                raise ValueError(
                    "Search node phase_telemetry must map phase names to "
                    "objects"
                )
            for metric_name, metric_value in phase_values.items():
                if (
                    not isinstance(metric_name, str)
                    or isinstance(metric_value, bool)
                    or not isinstance(metric_value, (int, float))
                    or not math.isfinite(float(metric_value))
                    or float(metric_value) < 0
                ):
                    raise ValueError(
                        "Search node phase_telemetry values must be finite "
                        "and non-negative"
                    )

        for name in ("build_fidelity", "eval_fidelity"):
            if name in values and values[name] not in FIDELITIES:
                raise ValueError(
                    f"Search node {name} must be one of {sorted(FIDELITIES)}"
                )
        promoted_from = values.get("promoted_from")
        if promoted_from is not None and (
            isinstance(promoted_from, bool)
            or not isinstance(promoted_from, int)
            or promoted_from < 0
        ):
            raise ValueError(
                "Search node promoted_from must be null or non-negative"
            )
        raw_attempts = values.get("evaluation_attempts", [])
        if not isinstance(raw_attempts, list):
            raise ValueError(
                "Search node evaluation_attempts must be a list"
            )
        values["evaluation_attempts"] = [
            attempt
            if isinstance(attempt, EvaluationAttempt)
            else EvaluationAttempt.from_dict(attempt)
            for attempt in raw_attempts
        ]

        from kapso.execution.iteration_evaluator import (
            normalize_metadata,
            normalize_metrics,
        )

        metrics, primary_metric = normalize_metrics(
            values.get("metrics", {}),
            values.get("primary_metric"),
        )
        values["metrics"] = metrics
        values["primary_metric"] = primary_metric
        values["external_evaluation_metadata"] = normalize_metadata(
            values.get("external_evaluation_metadata", {})
        )
        return cls(**values)
    
    def __str__(self) -> str:
        if self.had_error:
            return f"- Node {self.node_id} failed: {self.error_message[:100]}...\n  Solution: {self.solution[:200]}..."
        else:
            return (
                f"- Node {self.node_id} (score={self.score}):\n"
                f"  Solution: {self.solution[:200]}...\n"
                + (f"  Feedback: {self.feedback[:200]}...\n" if self.feedback else "")
            )


@dataclass
class ExperimentResult:
    """
    Result of a single experiment.
    
    DEPRECATED: Use SearchNode instead. Kept for backward compatibility.
    """
    node_id: int
    solution: str
    score: float
    branch_name: str
    had_error: bool
    error_message: str = ""
    output: str = ""
    detailed_output: str = ""
    feedbacks: str = ""
    embedding: List[float] = None
    evaluation_output: str = ""
    evaluation_script_path: str = ""
    evaluation_valid: bool = True
    evaluation_provenance: str = AGENT_GENERATED
    evaluation_integrity_error: str = ""
    code_diff: str = ""
    workspace_dir: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    primary_metric: Optional[str] = None
    external_evaluation_metadata: Dict[str, Any] = field(default_factory=dict)
    external_evaluation_error: str = ""
    
    def __str__(self) -> str:
        if self.had_error:
            return f"- Experiment with failed implementation error {self.error_message}. :\n  {self.solution} "
        else:
            return (
                f"- Experiment with final score {self.score} :\n # Solution : {self.solution}" 
                + (f"\n\n  # Runtime output: {self.output}" if self.output else "")
                + (f"\n\n  # Feedbacks: {self.feedbacks} \n" if self.feedbacks else "")
            )
    
    def get_embedding(self, llm: LLMBackend) -> List[float]:
        if self.embedding is None:
            self.embedding = llm.create_embedding(self.__str__())
        return self.embedding
    
    @classmethod
    def from_search_node(cls, node: SearchNode) -> "ExperimentResult":
        """Convert SearchNode to ExperimentResult for backward compatibility."""
        return cls(
            node_id=node.node_id,
            solution=node.solution,
            score=node.score or 0.0,
            branch_name=node.branch_name,
            had_error=node.had_error,
            error_message=node.error_message,
            output=node.agent_output,
            detailed_output=node.agent_output,
            feedbacks=node.feedback,
            evaluation_output=node.evaluation_output,
            evaluation_script_path=node.evaluation_script_path,
            evaluation_valid=node.evaluation_valid,
            evaluation_provenance=node.evaluation_provenance,
            evaluation_integrity_error=node.evaluation_integrity_error,
            code_diff=node.code_diff,
            workspace_dir=node.workspace_dir,
            metrics=dict(node.metrics),
            primary_metric=node.primary_metric,
            external_evaluation_metadata=dict(
                node.external_evaluation_metadata
            ),
            external_evaluation_error=node.external_evaluation_error,
        )


@dataclass 
class SearchStrategyConfig:
    """Configuration passed to search strategies."""
    problem_handler: ProblemHandler
    llm: LLMBackend
    coding_agent_config: CodingAgentConfig
    # Strategy-specific params (from YAML config)
    params: Dict[str, Any] = field(default_factory=dict)
    # Optional: start experiments from an existing local repo (copy/clone into workspace)
    initial_repo: Optional[str] = None
    # Optional: directories to copy into workspace
    eval_dir: Optional[str] = None
    data_dir: Optional[str] = None
    evaluation_manifest: Optional[Dict[str, str]] = None
    # Optional: FeedbackGenerator for generating feedback after each experiment
    feedback_generator: Optional["FeedbackGenerator"] = None
    # Goal string for feedback generation
    goal: str = ""


class SearchStrategy(ABC):
    """
    Abstract base class for experiment search strategies.
    
    Subclasses must implement:
    - run(): Execute one iteration of the search, returns SearchNode
    - get_experiment_history(): Return all experiments
    - get_best_experiment(): Return best experiment so far
    - checkout_to_best_experiment_branch(): Checkout to best solution
    - dump_state()/load_state(): Serialize versioned strategy state
    
    Shared infrastructure provided:
    - Workspace creation and management
    - RepoMemory bootstrap
    - Kapso directory setup (eval_dir, data_dir)
    - Optional legacy checkpoint hooks for trusted migration
    """
    
    WORKSPACE_FOLDER_BASE = 'tmp/search_strategy_workspace'
    
    def __init__(self, config: SearchStrategyConfig, workspace_dir: Optional[str] = None, import_from_checkpoint: bool = False):
        """
        Initialize search strategy.
        
        Args:
            config: SearchStrategyConfig with problem_handler, llm, coding_agent_config, params
            workspace_dir: Path to the workspace directory (optional)
            import_from_checkpoint: Whether to import state from checkpoint
        """
        self.problem_handler = config.problem_handler
        self.llm = config.llm
        self.params = config.params
        # The orchestrator's per-iteration budget view; strategies read it,
        # only the orchestrator writes budget state. The monotonic anchor
        # lets sequential phases inside one iteration discount time already
        # burned since the snapshot was taken.
        self.budget_snapshot: Optional["BudgetSnapshot"] = None
        self.budget_snapshot_monotonic: Optional[float] = None
        # The executive's granted workload profile for this iteration.
        self.fidelity_decision: Optional["FidelityDecision"] = None
        self.evaluation_provenance = (
            PROVIDED if config.eval_dir else AGENT_GENERATED
        )
        self.provided_evaluation_manifest: Dict[str, str] = {}
        self.provided_evaluation_fingerprint: Optional[str] = None
        if self.evaluation_provenance == PROVIDED:
            self.provided_evaluation_manifest = dict(
                config.evaluation_manifest
                if config.evaluation_manifest is not None
                else build_evaluation_manifest(config.eval_dir)
            )
            self.provided_evaluation_fingerprint = manifest_fingerprint(
                self.provided_evaluation_manifest
            )
        # Set by the orchestrator when an EvaluationMaintainer is active:
        # the registered evaluation tree is then enforced for EVERY
        # candidate, in every provenance mode, and candidates run the
        # registered command instead of building their own evaluation.
        self.registered_evaluation_manifest: Dict[str, str] = {}
        self.registered_evaluation_command: str = ""
        self.registered_evaluator_id: str = ""
        self.registered_subsample_seed: int = 0
        self.repo_memory_failure_policy = (
            RepoMemoryManager.normalize_failure_policy(
                self.params.get(
                    "repo_memory_failure_policy",
                    RepoMemoryManager.DEFAULT_FAILURE_POLICY,
                )
            )
        )
        self.repo_memory_max_retries = (
            RepoMemoryManager.normalize_max_retries(
                self.params.get(
                    "repo_memory_max_retries",
                    RepoMemoryManager.DEFAULT_MAX_RETRIES,
                )
            )
        )
        
        # Feedback generator and goal for generating feedback after experiments
        self.feedback_generator = config.feedback_generator
        self.goal = config.goal
        
        # Create experiment workspace with coding agent config
        if workspace_dir is None:
            self.workspace_dir = os.path.join(self.WORKSPACE_FOLDER_BASE, str(uuid.uuid4()))
        else:
            self.workspace_dir = workspace_dir
        self.workspace = ExperimentWorkspace(
            coding_agent_config=config.coding_agent_config,
            workspace_dir=self.workspace_dir,
            initial_repo=config.initial_repo,
            repo_memory_failure_policy=self.repo_memory_failure_policy,
            repo_memory_max_retries=self.repo_memory_max_retries,
            llm_backend=self.llm,
        )

        # Setup kapso directories (eval_dir -> kapso_evaluation/, data_dir -> kapso_datasets/)
        if not import_from_checkpoint:
            self._setup_kapso_directories(config.eval_dir, config.data_dir)

        # Ensure baseline RepoMemory exists in the workspace repo.
        if not import_from_checkpoint:
            self._initialize_repo_memory()

    # =========================================================================
    # Directory Setup
    # =========================================================================

    def _initialize_repo_memory(self) -> None:
        """Create baseline memory without making enrichment mandatory."""
        try:
            if self.workspace.is_seeded:
                RepoMemoryManager.bootstrap_baseline_model(
                    repo_root=self.workspace_dir,
                    llm=self.llm,
                    initial_repo=self.workspace.initial_repo,
                    max_retries=self.repo_memory_max_retries,
                )
            else:
                RepoMemoryManager.ensure_exists_in_worktree(
                    self.workspace_dir
                )
        except Exception as exc:
            if self.repo_memory_failure_policy == "fail":
                raise
            logger.warning(
                "RepoMemory bootstrap failed; continuing with the deterministic "
                "repository map only: %s: %s",
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            RepoMemoryManager.ensure_exists_in_worktree(
                self.workspace_dir,
                initial_repo=self.workspace.initial_repo,
            )

        self.workspace.repo.git.add([RepoMemoryManager.MEMORY_REL_PATH])
        if self.workspace.repo.is_dirty(untracked_files=True):
            self.workspace.repo.git.commit(
                "-m", "chore(kapso): add baseline repo memory"
            )
    
    def _setup_kapso_directories(
        self, 
        eval_dir: Optional[str], 
        data_dir: Optional[str]
    ) -> None:
        """
        Setup kapso_evaluation/ and kapso_datasets/ directories in workspace.
        
        Copies user-provided directories into the workspace repo so the agent
        has access to evaluation scripts and datasets.
        """
        workspace = self.workspace.workspace_dir
        dirs_created = []
        
        # Setup kapso_evaluation/
        kapso_eval = os.path.join(workspace, "kapso_evaluation")
        if eval_dir and os.path.exists(eval_dir):
            # The caller-provided suite is authoritative. Avoid silently
            # mixing it with evaluation files from a seeded repository.
            shutil.rmtree(kapso_eval, ignore_errors=True)
            shutil.copytree(eval_dir, kapso_eval)
            print("  Copied eval_dir to kapso_evaluation/")
        else:
            os.makedirs(kapso_eval, exist_ok=True)
        dirs_created.append("kapso_evaluation")
        
        # Setup kapso_datasets/
        kapso_data = os.path.join(workspace, "kapso_datasets")
        os.makedirs(kapso_data, exist_ok=True)
        if data_dir and os.path.exists(data_dir):
            shutil.copytree(data_dir, kapso_data, dirs_exist_ok=True)
            print("  Copied data_dir to kapso_datasets/")
        dirs_created.append("kapso_datasets")
        
        # Add placeholder files to empty directories so git tracks them
        for dir_name in dirs_created:
            dir_path = os.path.join(workspace, dir_name)
            if not os.listdir(dir_path):
                placeholder = os.path.join(dir_path, ".gitkeep")
                with open(placeholder, "w") as f:
                    f.write("# Placeholder to track empty directory\n")
        
        # Commit the directories to the workspace repo
        if self.evaluation_provenance == PROVIDED:
            # Seed repositories may ignore test/config suffixes. Caller-owned
            # evaluation files must still be present in every candidate clone.
            self.workspace.repo.git.add("-f", "kapso_evaluation")
            self.workspace.repo.git.add(["kapso_datasets"])
        else:
            self.workspace.repo.git.add(dirs_created)
        if self.workspace.repo.is_dirty(untracked_files=True):
            self.workspace.repo.git.commit("-m", "chore(kapso): setup evaluation and data directories")

        if self.evaluation_provenance == PROVIDED:
            copied_manifest = build_evaluation_manifest(kapso_eval)
            if copied_manifest != self.provided_evaluation_manifest:
                raise RuntimeError(
                    "Copied evaluation suite does not match its source "
                    "manifest"
                )
    
    # =========================================================================
    # Shared Helpers
    # =========================================================================

    def _get_code_diff(self, branch_name: str, parent_branch: str) -> str:
        """Get git diff between branch and parent."""
        try:
            diff = self.workspace.repo.git.diff(parent_branch, branch_name)
            return diff
        except Exception as e:
            print(f"[SearchStrategy] Warning: Could not get diff: {e}")
            return ""

    def observe_budget(self, snapshot: "BudgetSnapshot") -> None:
        """Store the orchestrator's read-only budget view for this iteration.

        Additive by design: strategies that ignore budgets inherit an inert
        attribute, and no run() signature changes across strategies.
        """
        self.budget_snapshot = snapshot
        self.budget_snapshot_monotonic = time.monotonic()

    def observe_fidelity(self, decision: "FidelityDecision") -> None:
        """Store the executive's granted profile for this iteration."""
        self.fidelity_decision = decision

    def dump_evaluation_integrity_state(self) -> Dict[str, Any]:
        """Return the provided-suite baseline stored with strategy state."""
        return {
            "provenance": getattr(
                self,
                "evaluation_provenance",
                AGENT_GENERATED,
            ),
            "manifest": dict(
                getattr(self, "provided_evaluation_manifest", {})
            ),
            "fingerprint": getattr(
                self,
                "provided_evaluation_fingerprint",
                None,
            ),
        }

    def load_evaluation_integrity_state(self, state: Any) -> None:
        """Validate persisted evaluation provenance against this invocation."""
        if state is None:
            return
        if not isinstance(state, dict):
            raise ValueError("Evaluation integrity state must be an object")
        provenance = state.get("provenance")
        manifest = state.get("manifest")
        fingerprint = state.get("fingerprint")
        if (
            not isinstance(provenance, str)
            or provenance not in VALID_PROVENANCE
        ):
            raise ValueError("Evaluation integrity provenance is invalid")
        if not isinstance(manifest, dict) or any(
            not isinstance(path, str) or not isinstance(digest, str)
            for path, digest in manifest.items()
        ):
            raise ValueError("Evaluation integrity manifest is invalid")
        expected_fingerprint = (
            manifest_fingerprint(manifest) if manifest else None
        )
        if fingerprint != expected_fingerprint:
            raise ValueError("Evaluation integrity fingerprint is invalid")
        if not hasattr(self, "evaluation_provenance"):
            self.evaluation_provenance = provenance
            self.provided_evaluation_manifest = dict(manifest)
            self.provided_evaluation_fingerprint = fingerprint
            return
        if provenance != self.evaluation_provenance:
            raise ValueError(
                "Evaluation integrity provenance changed on resume"
            )
        if manifest != self.provided_evaluation_manifest:
            raise ValueError("Provided evaluation suite changed on resume")

    def set_registered_evaluation(
        self,
        *,
        manifest: Dict[str, str],
        command: str,
        evaluator_id: str,
        subsample_seed: int,
    ) -> None:
        """Adopt a maintainer-registered evaluation as the enforced baseline."""
        self.registered_evaluation_manifest = dict(manifest)
        self.registered_evaluation_command = command
        self.registered_evaluator_id = evaluator_id
        self.registered_subsample_seed = subsample_seed

    def enforce_evaluation_integrity(self, node: SearchNode) -> bool:
        """Reject a candidate that changed the evaluator it was scored by.

        With a maintainer-registered evaluation, the registered manifest is
        enforced for every candidate regardless of provenance; otherwise the
        caller-provided manifest is enforced and agent-generated evaluation
        remains unchecked (the pre-maintainer regime).
        """
        node._evaluation_integrity_checked = True
        node.evaluation_provenance = self.evaluation_provenance
        node.evaluation_integrity_error = ""
        if self.registered_evaluation_manifest:
            enforced_manifest = self.registered_evaluation_manifest
        elif self.evaluation_provenance == AGENT_GENERATED:
            return True
        else:
            enforced_manifest = self.provided_evaluation_manifest

        try:
            with self.workspace.materialize_ref(
                node.branch_name
            ) as candidate_dir:
                evaluation_dir = os.path.join(
                    candidate_dir,
                    "kapso_evaluation",
                )
                report = verify_evaluation_tree(
                    evaluation_dir,
                    enforced_manifest,
                )
        except Exception as exc:
            node.evaluation_integrity_error = (
                f"{type(exc).__name__}: {exc}"
            )
        else:
            node.evaluation_integrity_error = report.error

        if node.evaluation_integrity_error:
            node.evaluation_valid = False
            node.score = None
            node.should_stop = False
            node.feedback = node.evaluation_integrity_error
            return False
        return True

    # =========================================================================
    # Abstract Methods - Must be implemented by subclasses
    # =========================================================================
    
    @abstractmethod
    def run(self, context: ContextData, budget_progress: float = 0.0) -> Optional[SearchNode]:
        """
        Execute one iteration of the search strategy.
        
        Args:
            context: Problem context, KG results, experiment history
            budget_progress: 0-100 indicating budget consumed
            
        Returns:
            SearchNode with solution, evaluation_output, feedback, should_stop
        """
        pass
    
    @abstractmethod
    def get_experiment_history(self, best_last: bool = False) -> List[SearchNode]:
        """
        Get all experiment results.
        
        Args:
            best_last: If True, sort by score (best last)
            
        Returns:
            List of SearchNode
        """
        pass
    
    @abstractmethod
    def get_best_experiment(self) -> Optional[SearchNode]:
        """Get the best experiment result so far."""
        pass
    
    @abstractmethod
    def checkout_to_best_experiment_branch(self) -> Optional[str]:
        """Checkout and return the best experiment branch, if one exists."""
        pass

    def dump_state(self) -> Dict[str, Any]:
        """Return JSON-compatible strategy state for a run checkpoint."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support resumable state"
        )

    def load_state(self, state: Dict[str, Any]) -> None:
        """Restore state produced by :meth:`dump_state`."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support resumable state"
        )

