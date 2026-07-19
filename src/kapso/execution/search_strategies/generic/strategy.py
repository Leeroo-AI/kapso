"""Evidence-directed Generic search with durable ideation and execution links."""

import glob
import hashlib
import json
import logging
import math
import os
import re
import shutil
import signal
import time
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from kapso.execution.search_strategies.base import (
    SearchStrategy,
    SearchStrategyConfig,
    SearchNode,
)
from kapso.execution.search_strategies.factory import register_strategy
from kapso.execution.fidelity import (
    PROFILE_VALIDATE,
    ComparabilityClass,
    EvaluationAttempt,
    FidelityDecision,
    project_score,
    select_committed_candidate,
)
from kapso.execution.evaluation_integrity import verify_data_manifest
from kapso.execution.evaluation_maintainer.maintainer import (
    MANIFEST_MARKER,
    evaluation_command,
    parse_manifest_line,
)
import shlex
import subprocess
from kapso.execution.memories.repo_memory import RepoMemoryManager
from kapso.execution.memories.experiment_memory import (
    ExperimentHistoryStore,
    ExperimentRecord,
)
from kapso.core.prompt_loader import load_prompt, render_prompt
from kapso.execution.search_strategies.generic.difficulties_generator import (
    generate_technical_difficulties,
)
from kapso.execution.search_strategies.generic.ideation.evaluator_evidence import (
    build_evaluator_evidence_writeback,
)
from kapso.execution.search_strategies.generic.ideation import (
    AnalyzerSettings,
    BatchStatus,
    CampaignAction,
    CampaignEvidenceBuilder,
    CandidateAnalyzer,
    CandidateGenerator,
    CandidateGeneratorSettings,
    CandidateSelector,
    CodingAgentRunnerSettings,
    EmbeddingSettings,
    EvidenceAuthor,
    EvaluationAttemptInput,
    ExperimentInput,
    EvidenceSettings,
    GapPrioritySettings,
    GenerationMemberSettings,
    IdeaArchive,
    IDEA_ARCHIVE_SCHEMA,
    IdeationCapacityView,
    IdeationEngine,
    IdeationEngineTelemetry,
    ObjectiveDirection,
    OpenAIEmbeddingProvider,
    OperatorSettings,
    ParentPlan,
    ParentPlanKind,
    ResolvedParentSnapshot,
    SubprocessCodingAgentCallRunner,
    build_idea_outcome,
    content_identifier,
    new_identifier,
)

if TYPE_CHECKING:
    from kapso.execution.search_strategies.generic import FeedbackGenerator

logger = logging.getLogger(__name__)

GENERIC_SEARCH_STATE_SCHEMA = "kapso.generic_search.v3"

_GENERIC_SEARCH_STATE_FIELDS = {
    "schema",
    "campaign_id",
    "idea_archive_schema",
    "archive_revision",
    "active_batch_id",
    "node_history",
    "iteration_count",
    "previous_errors",
    "evaluation_integrity",
    "scores_evaluator_id",
    "evaluator_transition",
}

# Enforcement mechanic (mirrors the coding-agent adapter's deadline grace):
# time granted between SIGTERM and SIGKILL when a frame run overruns.
_FRAME_RUN_KILL_GRACE_SECONDS = 2.0

_IDEATION_CONFIG_KEYS = {
    "archive_path",
    "evidence",
    "gaps",
    "operators",
    "coding_agents",
    "embeddings",
    "analyzer",
}

_GENERIC_PARAM_KEYS = {
    "ideation",
    "implementation_model",
    "implementation_timeout",
    "implementation_gates",
    "auth_mode",
    "aws_region",
    "effort",
    "env_strip",
    "session_env_defaults",
    "gate_failure_policy",
    "registered_evaluation_archive_glob",
    "repo_memory_failure_policy",
    "repo_memory_max_retries",
}

# The implementation output contract's terminal tags: a result event
# carrying ALL of these means the session declared itself complete (drives
# the adapter's linger-reap and truthful end-mode classification).
IMPLEMENTATION_COMPLETION_MARKERS = ["</score>", "</technical_difficulties>"]

# Byte-identical to the pre-maintainer template text: rendered whenever no
# maintainer-registered evaluation exists, keeping default prompts unchanged.
DEFAULT_EVALUATION_INSTRUCTIONS = """You MUST build and run evaluation in `kapso_evaluation/` directory:

1. **Create evaluation script**: `kapso_evaluation/evaluate.py` (or similar)
2. **Evaluation should**:
   - Test your solution against the goal criteria
   - Output a clear score or success/failure indication
   - Be fair and actually test what it claims to test
   - NOT be hardcoded or trivially pass

3. **Run the evaluation**: Execute your evaluation script and capture output.

4. **Retry on crash**: If evaluation crashes, fix the issue and retry (max 3 attempts)."""


@register_strategy("generic")
class GenericSearch(SearchStrategy):
    """Evidence-directed ideation with linked, resumable execution."""

    def __init__(
        self,
        config: SearchStrategyConfig,
        workspace_dir: Optional[str] = None,
        import_from_checkpoint: bool = False,
    ):
        """Initialize generic search strategy."""
        if not isinstance(config.params, dict):
            raise ValueError("generic search params must be an object")
        unknown_params = sorted(set(config.params) - _GENERIC_PARAM_KEYS)
        if unknown_params:
            raise ValueError(
                "generic search params contain unknown fields: "
                + ", ".join(unknown_params)
            )
        raw_ideation_config = config.params.get("ideation")
        if (
            not isinstance(raw_ideation_config, dict)
            or set(raw_ideation_config) != _IDEATION_CONFIG_KEYS
        ):
            raise ValueError("generic search ideation configuration is invalid")
        super().__init__(config, workspace_dir, import_from_checkpoint)
        self.ideation_config = raw_ideation_config
        self.ideation_campaign_id = (
            None if import_from_checkpoint else new_identifier("campaign")
        )
        self.idea_archive: Optional[IdeaArchive] = None
        self.active_batch_id: Optional[str] = None

        if self.params.get("auth_mode") is not None:
            self._claude_auth_settings = {"auth_mode": self.params["auth_mode"]}
        else:
            self._claude_auth_settings = {"auth_mode": "bedrock"}
        self.aws_region = self.params.get("aws_region", "us-east-1")
        # Optional reasoning effort for implementation sessions.
        self.session_effort = self.params.get("effort")
        # Environment controls for implementation-agent credential containment.
        self.env_strip = list(self.params.get("env_strip", []))
        self.env_defaults = dict(self.params.get("session_env_defaults", {}))
        # Durable-archive recovery root for the registered evaluation (glob
        # of run archive parents, e.g. "tmp/relbench/*/runs"). None disables
        # archive recovery; the live-process wait still applies.
        self.registered_evaluation_archive_glob = self.params.get(
            "registered_evaluation_archive_glob"
        )
        # Config params for implementation
        self.implementation_model = self.params.get(
            "implementation_model", "us.anthropic.claude-opus-4-5-20251101-v1:0"
        )
        self.implementation_timeout = self.params.get("implementation_timeout", 600)
        self.gate_failure_policy = self.params.get("gate_failure_policy", "warn")
        self.implementation_gates = self.params.get(
            "implementation_gates", ["research", "repo_memory", "leeroopedia"]
        )
        # State
        self.node_history: List[SearchNode] = []
        self.iteration_count = 0
        # Which evaluator version node.score projections currently reflect,
        # and the in-flight evaluator transition (pending until the bridge
        # evaluation anchors the frontier on the new version).
        self.scores_evaluator_id: str = ""
        self.evaluator_transition: Optional[Dict[str, str]] = None

        # Error tracking for implementation feedback
        self.previous_errors: List[str] = []
        self.recent_error_count = 3  # Number of recent errors to include in prompts

        print(f"[GenericSearch] Initialized:")
        print(f"  - implementation_model: {self.implementation_model}")
        print(f"  - auth: {self._claude_auth_settings}")
        print(f"  - implementation_gates: {self.implementation_gates}")
        print(f"  - gate_failure_policy: {self.gate_failure_policy}")
        print(
            f"  - feedback_generator: {'configured' if self.feedback_generator else 'not configured'}"
        )

        # Initialize workspace with empty main file only for empty workspaces.
        # If the workspace is seeded from an existing repo, we must not overwrite it.
        if workspace_dir is None and not self.workspace.is_seeded:
            self._initialize_workspace()

        if not import_from_checkpoint:
            self._ensure_idea_archive()

    def _initialize_workspace(self) -> None:
        """Create initial empty main file."""
        session = self.workspace.create_experiment_session(
            branch_name=self.workspace.get_current_branch()
        )
        session.generate_code(
            f"<problem>\n{self.problem_handler.get_problem_context()}\n</problem>\n\n"
            + "Create an empty main with a main() function placeholder. No comments."
        )
        self.workspace.finalize_session(session)
        self.workspace.repo.git.stash()

    def _workspace_config_path(self, configured_path: str) -> Path:
        if not isinstance(configured_path, str) or not configured_path.strip():
            raise ValueError("ideation workspace path must be non-empty")
        relative = Path(configured_path)
        if relative.is_absolute():
            raise ValueError("ideation workspace paths must be relative")
        workspace_root = Path(self.workspace_dir).resolve()
        resolved = (workspace_root / relative).resolve()
        if not resolved.is_relative_to(workspace_root):
            raise ValueError("ideation workspace path escapes the workspace")
        return resolved

    def _ensure_idea_archive(self) -> IdeaArchive:
        if self.ideation_campaign_id is None:
            raise ValueError("ideation campaign identity is not loaded")
        archive_path = self._workspace_config_path(self.ideation_config["archive_path"])
        if self.idea_archive is None:
            self.idea_archive = IdeaArchive(
                archive_path,
                self.ideation_campaign_id,
            )
        elif (
            self.idea_archive.path != archive_path
            or self.idea_archive.campaign_id != self.ideation_campaign_id
        ):
            raise ValueError("loaded idea archive identity changed")
        return self.idea_archive

    def _build_ideation_engine(self) -> IdeationEngine:
        archive = self._ensure_idea_archive()
        coding_config = self.ideation_config["coding_agents"]
        if not isinstance(coding_config, dict) or set(coding_config) != {
            "artifact_path",
            "termination_grace_seconds",
            "evidence_author",
            "generator",
            "selector",
        }:
            raise ValueError("ideation coding-agent configuration is invalid")
        runner = self._build_ideation_call_runner(coding_config)
        embedding_settings = EmbeddingSettings.from_dict(
            self.ideation_config["embeddings"]
        )
        embedding_provider = (
            OpenAIEmbeddingProvider(embedding_settings)
            if embedding_settings.enabled
            else None
        )
        evidence_values = dict(self.ideation_config["evidence"])
        evidence_values["evaluator_id"] = (
            self.registered_evaluator_id or "unregistered_evaluator"
        )
        evidence_values["comparable_seed"] = self.registered_subsample_seed
        return IdeationEngine(
            archive=archive,
            evidence_builder=CampaignEvidenceBuilder(
                EvidenceSettings.from_dict(evidence_values)
            ),
            operator_settings=OperatorSettings.from_dict(
                self.ideation_config["operators"]
            ),
            gap_priority_settings=GapPrioritySettings.from_dict(
                self.ideation_config["gaps"]
            ),
            generator=CandidateGenerator(
                runner,
                CandidateGeneratorSettings.from_dict(coding_config["generator"]),
            ),
            analyzer=CandidateAnalyzer(
                AnalyzerSettings.from_dict(self.ideation_config["analyzer"]),
                embedding_provider=embedding_provider,
            ),
            selector=CandidateSelector(
                runner,
                GenerationMemberSettings.from_dict(coding_config["selector"]),
            ),
        )

    def _build_ideation_call_runner(
        self,
        coding_config: Dict[str, Any],
    ) -> SubprocessCodingAgentCallRunner:
        return SubprocessCodingAgentCallRunner(
            CodingAgentRunnerSettings(
                artifact_root=str(
                    self._workspace_config_path(coding_config["artifact_path"])
                ),
                termination_grace_seconds=coding_config["termination_grace_seconds"],
            )
        )

    def _build_ideation_evidence_author(self) -> EvidenceAuthor:
        coding_config = self.ideation_config["coding_agents"]
        return EvidenceAuthor(
            self._build_ideation_call_runner(coding_config),
            GenerationMemberSettings.from_dict(coding_config["evidence_author"]),
        )

    def _ideation_capacity_view(self) -> IdeationCapacityView:
        snapshot = self.budget_snapshot
        decision = self.fidelity_decision
        if snapshot is None or decision is None:
            raise ValueError("generic ideation requires budget and fidelity authority")
        evidence_config = self.ideation_config["evidence"]
        comparable_evaluation = decision.eval_fidelity == evidence_config[
            "comparable_fidelity"
        ] and math.isclose(
            decision.eval_fraction,
            evidence_config["comparable_fraction"],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        can_start = not snapshot.exhausted or decision.reserve_run
        remaining_after_reserve = snapshot.remaining_after_reserve
        preserves_reserve = decision.reserve_run or (
            remaining_after_reserve is None or remaining_after_reserve > 0
        )
        can_run_granted_evaluation = decision.profile in {"probe", "full"}
        authority = {
            "iteration_index": snapshot.iteration_index,
            "max_iterations": snapshot.max_iterations,
            "remaining_seconds": snapshot.remaining_seconds,
            "remaining_after_reserve_seconds": snapshot.remaining_after_reserve,
            "remaining_usd": snapshot.remaining_usd,
            "fidelity_profile": decision.profile,
            "build_fidelity": decision.build_fidelity,
            "eval_fidelity": decision.eval_fidelity,
            "eval_fraction": decision.eval_fraction,
            "target_node_id": decision.target_node_id,
            "reserve_run": decision.reserve_run,
            "deadline_seconds": decision.deadline_seconds,
            "can_start_complete_action": can_start,
            "can_run_granted_evaluation": can_run_granted_evaluation,
            "can_run_comparable_evaluation": comparable_evaluation,
            "preserves_finalization_reserve": preserves_reserve,
        }
        digest = hashlib.sha256(
            json.dumps(
                authority,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        return IdeationCapacityView(
            capacity_snapshot_id=content_identifier(
                "capacity_snapshot",
                digest,
            ),
            **authority,
        )

    def _ideation_experiment_inputs(self) -> Tuple[ExperimentInput, ...]:
        experiments = []
        for node in self.node_history:
            if node.idea_id is None or node.selection_batch_id is None:
                raise ValueError("generic experiment history contains an unlinked node")
            if not node.started_at:
                raise ValueError("generic experiment is missing its start timestamp")
            attempts = tuple(
                EvaluationAttemptInput(
                    evaluator_id=attempt.evaluator_id,
                    fidelity=attempt.fidelity,
                    fraction=attempt.fraction,
                    seed=attempt.seed,
                    score=attempt.score,
                    duration_seconds=attempt.duration_seconds,
                )
                for attempt in node.evaluation_attempts
            )
            experiments.append(
                ExperimentInput(
                    node_id=node.node_id,
                    idea_id=node.idea_id,
                    selection_batch_id=node.selection_batch_id,
                    parent_node_id=node.parent_node_id,
                    proposal=node.solution,
                    score=node.score,
                    evaluation_valid=node.evaluation_valid,
                    had_error=node.had_error,
                    recoverable_error=node.recoverable_error,
                    build_fidelity=node.build_fidelity,
                    attempts=attempts,
                    feedback=node.feedback,
                    technical_difficulty=(node.technical_difficulties or None),
                    created_at=node.started_at,
                )
            )
        return tuple(experiments)

    def _resolve_ideation_parent(
        self,
        parent_plan: ParentPlan,
    ) -> ResolvedParentSnapshot:
        if parent_plan.kind == ParentPlanKind.BASELINE:
            branch_name = "main"
            node_id = None
        else:
            node_id = parent_plan.experiment_node_id
            if node_id is None:
                raise ValueError("experiment parent plan requires a node id")
            nodes = {node.node_id: node for node in self.node_history}
            if node_id not in nodes:
                raise ValueError(f"parent experiment node does not exist: {node_id}")
            branch_name = nodes[node_id].branch_name
        if not branch_name:
            raise ValueError("parent experiment has no committed branch")
        commit_sha = self.workspace.repo.commit(branch_name).hexsha
        return ResolvedParentSnapshot(
            node_id=node_id,
            branch_name=branch_name,
            git_ref=commit_sha,
            materialized_ref=commit_sha,
            diff_base_ref=commit_sha,
            feedback_base_ref=commit_sha,
        )

    def _materialize_ideation_parent(self, parent: ResolvedParentSnapshot):
        return self.workspace.materialize_ref(parent.materialized_ref)

    def _assert_parent_snapshot_current(
        self,
        parent: ResolvedParentSnapshot,
    ) -> None:
        current_sha = self.workspace.repo.commit(parent.branch_name).hexsha
        if current_sha != parent.git_ref:
            raise ValueError("selected parent branch changed after ideation")

    @staticmethod
    def _ideation_phase_telemetry(telemetry) -> Dict[str, float]:
        duration = telemetry.coding_agent_duration_seconds
        if telemetry.embedding is not None:
            duration += telemetry.embedding.duration_seconds
        return {
            "cost_usd": telemetry.known_coding_agent_cost_usd,
            "duration_seconds": duration,
            "coding_agent_call_count": float(telemetry.coding_agent_call_count),
            "unpriced_coding_agent_call_count": float(
                telemetry.unpriced_coding_agent_call_count
            ),
        }

    @staticmethod
    def _accumulate_phase_telemetry(
        node: SearchNode,
        phase_name: str,
        increments: Dict[str, float],
    ) -> None:
        if not isinstance(phase_name, str) or not phase_name:
            raise ValueError("phase telemetry name must be a non-empty string")
        phase = dict(node.phase_telemetry.get(phase_name, {}))
        for name, value in increments.items():
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(
                    f"phase telemetry increment {phase_name}.{name} is invalid"
                )
            current = phase.get(name, 0.0)
            if (
                not isinstance(current, (int, float))
                or isinstance(current, bool)
                or not math.isfinite(current)
                or current < 0
            ):
                raise ValueError(
                    f"existing phase telemetry {phase_name}.{name} is invalid"
                )
            phase[name] = current + value
        node.phase_telemetry[phase_name] = phase

    @classmethod
    def _add_evidence_author_telemetry(cls, node: SearchNode, call) -> None:
        increments = {
            "cost_usd": 0.0 if call.cost_usd is None else call.cost_usd,
            "duration_seconds": call.duration_seconds,
            "coding_agent_call_count": 1.0,
            "unpriced_coding_agent_call_count": (1.0 if call.cost_usd is None else 0.0),
        }
        if call.input_tokens is not None:
            increments["input_tokens"] = float(call.input_tokens)
        if call.output_tokens is not None:
            increments["output_tokens"] = float(call.output_tokens)
        cls._accumulate_phase_telemetry(node, "evidence_author", increments)

    def _author_ideation_evidence(
        self,
        node: SearchNode,
        problem_statement: str,
    ) -> None:
        archive = self._ensure_idea_archive()
        idea = archive.get_idea(node.idea_id)
        current_commit_sha = self.workspace.repo.commit(node.branch_name).hexsha
        with self.workspace.materialize_ref(node.branch_name) as materialized:
            authored = self._build_ideation_evidence_author().author(
                problem_statement=problem_statement,
                idea=idea,
                node=node,
                archive_state=archive.state,
                workspace=str(materialized),
                current_commit_sha=current_commit_sha,
                evaluator_id=self.registered_evaluator_id,
            )
        metadata = dict(node.external_evaluation_metadata)
        metadata.update(authored.metadata)
        node.external_evaluation_metadata = metadata
        self._add_evidence_author_telemetry(node, authored.call)

    @staticmethod
    def _should_author_ideation_evidence(node: SearchNode) -> bool:
        return (
            not node.had_error and not node.recoverable_error and node.evaluation_valid
        )

    def run(self, context: str, budget_progress: float = 0.0) -> SearchNode:
        """
        Execute one iteration of generic search.

        Node lifecycle:
        1. Build evidence, generate/analyze/select ideas, and persist the decision
        2. Link the selected idea to an execution node
        3. Implement and evaluate on the frozen parent
        4. Return the node for experiment-memory and outcome finalization

        Args:
            context: Complete problem statement for this iteration
            budget_progress: Budget progress percentage (0-100)

        Returns:
            SearchNode with solution, evaluation_output, feedback, should_stop
        """
        self.iteration_count += 1
        print(
            f"\n[GenericSearch] Iteration {self.iteration_count}, budget: {budget_progress:.1f}%"
        )

        # An eval-only VALIDATE grant short-circuits the whole lifecycle:
        # no ideation, no implementation — one full-fidelity measurement of
        # an existing artifact, appended to its node.
        decision = self.fidelity_decision
        if decision is not None and decision.profile == PROFILE_VALIDATE:
            return self._run_validate(decision)

        if not isinstance(context, str) or not context.strip():
            raise ValueError("generic search context must be a non-empty string")
        problem = context

        iteration_started_monotonic = time.monotonic()
        iteration_started_at = datetime.now(timezone.utc).isoformat()

        resume_batch_id = None
        if self.active_batch_id is not None:
            active_batch = self._ensure_idea_archive().get_batch(self.active_batch_id)
            if active_batch.status in {
                BatchStatus.PLANNED,
                BatchStatus.GENERATED,
                BatchStatus.ANALYZED,
                BatchStatus.SELECTED,
            }:
                resume_batch_id = active_batch.batch_id

        ideation_result = self._build_ideation_engine().run(
            campaign_id=self.ideation_campaign_id,
            iteration_index=self.iteration_count - 1,
            problem_statement=problem,
            objective_direction=(
                ObjectiveDirection.MAXIMIZE
                if self.problem_handler.maximize_scoring
                else ObjectiveDirection.MINIMIZE
            ),
            experiments=self._ideation_experiment_inputs(),
            capacity=self._ideation_capacity_view(),
            selector_workspace=self.workspace_dir,
            parent_resolver=self._resolve_ideation_parent,
            parent_materializer=self._materialize_ideation_parent,
            generated_at=iteration_started_at,
            resume_batch_id=resume_batch_id,
        )
        if ideation_result.action == CampaignAction.FINALIZE:
            deliverable = self.get_deliverable_experiment()
            if deliverable is None:
                raise ValueError("finalization requires a delivery incumbent")
            deliverable.should_stop = True
            deliverable.feedback = "; ".join(
                reason.statement
                for reason in ideation_result.directive.decision.reasons
            )
            return deliverable
        selected_idea = ideation_result.selected_idea
        parent = ideation_result.resolved_parent
        if selected_idea is None or parent is None or ideation_result.batch_id is None:
            raise ValueError("executable ideation result is incomplete")
        self.active_batch_id = ideation_result.batch_id
        solution = selected_idea.proposal
        print(f"[GenericSearch] Generated solution ({len(solution)} chars)")

        if ideation_result.action == CampaignAction.RECOVER:
            if selected_idea.experiment_node_id is None:
                raise ValueError("recovery idea is not linked to an experiment")
            node = self.node_history[selected_idea.experiment_node_id]
            if (
                node.idea_id != selected_idea.idea_id
                or node.selection_batch_id != ideation_result.batch_id
            ):
                raise ValueError("recovery node and idea provenance disagree")
            branch_name = node.branch_name
            node.execution_revision += 1
            node.should_stop = False
            node.had_error = False
            node.recoverable_error = False
            node.error_message = ""
            node.evaluation_valid = True
            node.evaluation_integrity_error = ""
            node.score = None
            prior_duration_seconds = (
                0.0 if node.duration_seconds is None else node.duration_seconds
            )
        else:
            node = SearchNode(
                node_id=len(self.node_history),
                parent_node_id=parent.node_id,
                idea_id=selected_idea.idea_id,
                selection_batch_id=ideation_result.batch_id,
                solution=solution,
                workspace_dir=self.workspace_dir,
            )
            node.started_at = iteration_started_at
            branch_name = f"generic_exp_{node.node_id}"
            self._ensure_idea_archive().link_experiment(
                selected_idea.idea_id,
                node.node_id,
                ideation_result.batch_id,
                expected_revision=self._ensure_idea_archive().revision,
            )
            prior_duration_seconds = 0.0
        self._accumulate_phase_telemetry(
            node,
            "ideation",
            self._ideation_phase_telemetry(ideation_result.telemetry),
        )
        if decision is not None:
            node.build_fidelity = decision.build_fidelity
            node.eval_fidelity = decision.eval_fidelity
            if decision.profile == "full":
                node.promoted_from = decision.target_node_id

        # Step 2: Implement - developer agent handles everything
        self._assert_parent_snapshot_current(parent)
        print(
            f"[GenericSearch] Implementing on branch: {branch_name} "
            f"(from {parent.branch_name})"
        )

        self._last_implementation_success = None
        self._last_implementation_error = ""
        agent_output, implementation_telemetry = self._implement(
            solution=solution,
            problem=problem,
            branch_name=branch_name,
            parent_branch_name=parent.branch_name,
        )
        if not isinstance(self._last_implementation_success, bool):
            raise ValueError("implementation did not report its completion status")
        node.had_error = not self._last_implementation_success
        node.error_message = self._last_implementation_error
        node.recoverable_error = node.had_error
        self._accumulate_phase_telemetry(
            node,
            "implementation",
            implementation_telemetry,
        )

        # Update node with implementation results
        node.branch_name = branch_name
        if ideation_result.action == CampaignAction.IDEATE:
            node.parent_branch_name = parent.branch_name
        node.implementation_base_ref = parent.git_ref
        if ideation_result.action == CampaignAction.IDEATE:
            node.diff_base_ref = parent.diff_base_ref
            node.feedback_base_ref = parent.feedback_base_ref
        node.agent_output = agent_output
        node.code_diff = self._get_code_diff(branch_name, node.diff_base_ref)

        # Step 3: successful sessions must satisfy the current XML contract.
        if node.had_error:
            node.evaluation_output = agent_output
        else:
            agent_result = self._extract_agent_result(agent_output)
            node.code_changes_summary = agent_result.get("code_changes_summary", "")
            node.evaluation_script_path = agent_result.get("evaluation_script_path", "")
            node.technical_difficulties = agent_result.get("technical_difficulties", "")
            node.evaluation_output = agent_result.get("evaluation_output", agent_output)
            # Score from agent result (may be overridden by feedback generator)
            if agent_result.get("score") is not None:
                node.score = float(agent_result.get("score", 0.0))
            print("[GenericSearch] Extracted implementation result from XML")

        # Step 3b: the implementor is the primary author of
        # technical_difficulties; the fallback reconstructs it when the tag
        # is missing. Purely mechanical trigger — never score/outcome-based.
        self._ensure_technical_difficulties(node)

        # Step 3c: inject a manifest recovered from the durable run archive
        # (stashed by the pre-finalize teardown guard) so the score of
        # record survives a session that died before printing it.
        recovered = getattr(self, "_recovered_manifest_line", None)
        if recovered and self._manifest_score_of_record(node) is None:
            node.evaluation_output = (node.evaluation_output or "") + "\n" + recovered
            self._recovered_manifest_line = None

        # Step 4: Verify provided evaluation files before accepting any score
        # or feedback derived from them.
        if node.had_error:
            node.evaluation_valid = False
            node.score = None
            node.feedback = node.error_message
        elif self.enforce_evaluation_integrity(node):
            self._generate_feedback(node)
            self._record_evaluation_attempt(node)
        else:
            print(
                "[GenericSearch] Rejected invalid provided evaluation: "
                f"{node.evaluation_integrity_error}"
            )

        if self._should_author_ideation_evidence(node):
            self._author_ideation_evidence(node, problem)

        # Stamp iteration totals: wall-clock for the whole iteration, spend as
        # the sum of attributed phase costs.
        node.duration_seconds = (
            prior_duration_seconds + time.monotonic() - iteration_started_monotonic
        )
        node.cost_usd = sum(
            phase.get("cost_usd", 0.0) for phase in node.phase_telemetry.values()
        )

        if ideation_result.action == CampaignAction.IDEATE:
            self.node_history.append(node)

        print(
            f"[GenericSearch] ✓ Node {node.node_id} completed: score={node.score}, should_stop={node.should_stop}"
        )

        return node

    def _implement(
        self,
        solution: str,
        problem: str,
        branch_name: str,
        parent_branch_name: str = "main",
    ) -> Tuple[str, Dict[str, float]]:
        """
        Implementation using Claude Code with MCP gates (code, research).

        Overrides base class to use Claude Code with Bedrock and MCP gates
        instead of the default coding agent from config.

        Args:
            solution: Solution description to implement
            problem: Problem description
            branch_name: Git branch for this experiment
            parent_branch_name: Parent branch to inherit code from
        Returns:
            Tuple of (agent output string, phase telemetry with cost/duration)
        """
        from kapso.execution.coding_agents.base import CodingAgentConfig
        from kapso.execution.coding_agents.adapters.claude_code_agent import (
            ClaudeCodeCodingAgent,
        )
        from kapso.gated_mcp import get_mcp_config
        from kapso.execution.memories.repo_memory.observation import (
            extract_repo_memory_sections_consulted,
        )

        # Create experiment session (handles git branching)
        session = self.workspace.create_experiment_session(
            branch_name, parent_branch_name, llm=self.llm
        )

        # A maintainer-registered evaluation is versioned on the workspace
        # root, but sessions inherit their parent branch's tree — which may
        # predate a re-registration. Frame-sync the registered tree in so
        # every candidate runs (and is integrity-checked against) the head.
        if self.registered_evaluation_manifest:
            self._sync_registered_evaluation(session.session_folder)

        # 1. Load RepoMemory
        repo_memory_doc = RepoMemoryManager.ensure_exists_in_worktree(
            session.session_folder
        )
        repo_memory_brief = RepoMemoryManager.render_summary_and_toc(
            repo_memory_doc, max_chars=2500
        )

        # 2. Get MCP config for code + research + repo_memory gates (not idea)
        mcp_servers, mcp_tools = get_mcp_config(
            gates=self.implementation_gates,
            repo_root=session.session_folder,
            include_base_tools=False,
            gate_failure_policy=self.gate_failure_policy,
        )

        # 3. Build full tool set for implementation (includes Write, Edit)
        # Bash is kept for running evaluation scripts, not for repo_memory access
        implementation_allowed_tools = [
            "Read",
            "Write",
            "Edit",
            "Bash",
            *[t for t in mcp_tools if t.startswith("mcp__")],
        ]

        logger.info(
            f"[GenericSearch] Implementation tools: {implementation_allowed_tools}"
        )

        # 4. Configure Claude Code for implementation
        config = CodingAgentConfig(
            agent_type="claude_code",
            model=self.implementation_model,
            debug_model=self.implementation_model,
            agent_specific={
                **self._claude_auth_settings,
                "env_strip": self.env_strip,
                "env_defaults": self.env_defaults,
                "aws_region": self.aws_region,
                "mcp_servers": mcp_servers,
                "allowed_tools": implementation_allowed_tools,
                "timeout": self._clamped_timeout(self.implementation_timeout),
                "streaming": True,
                "effort": self.session_effort,
                # Per-session process record: raw stream-json events land
                # here as they arrive, so a killed session still leaves its
                # forensics behind (feeds the difficulties fallback).
                "stream_artifact_path": self._session_stream_path(branch_name),
                # Declared-completion contract: lets the adapter reap a CLI
                # that delivered its full final report but lingers alive.
                "completion_markers": IMPLEMENTATION_COMPLETION_MARKERS,
            },
        )

        # 5. Build implementation prompt
        repo_memory_detail_access_instructions = (
            "For detailed section content (architecture, gotchas, invariants, etc.),\n"
            'use the MCP tool: `get_repo_memory_section(section_id="core.architecture")`\n'
            "Available sections: core.architecture, core.entrypoints, core.where_to_edit, core.invariants, core.testing, core.gotchas, core.dependencies\n"
            "Fallback: open `.kapso/repo_memory.json` and read `book.sections[section_id]`."
        )

        prompt = self._build_implementation_prompt(
            solution=solution,
            problem=problem,
            branch_name=branch_name,
            repo_memory_brief=repo_memory_brief,
            repo_memory_detail_access_instructions=repo_memory_detail_access_instructions,
            previous_errors="\n".join(
                str(e) for e in self.previous_errors[-self.recent_error_count :]
            ),
        )

        # 6. Run Claude Code for implementation
        print(f"[GenericSearch] Running Claude Code implementation...")
        agent = ClaudeCodeCodingAgent(config)
        agent.initialize(session.session_folder)

        phase_started = time.monotonic()
        phase_cost = 0.0
        try:
            self._last_session_started_ts = time.time()
            result = agent.generate_code(prompt)
            self._last_implementation_success = result.success
            self._last_implementation_error = result.error or ""
            phase_cost = agent.get_cumulative_cost()
            agent_output = result.output if result.output else ""

            # Ground truth about HOW the session ended, for the feedback
            # judge (run #8: a self-inflicted SIGTERM was misdiagnosed as
            # the time limit, so the footgun was never named).
            meta = result.metadata or {}
            if meta.get("completed_reaped"):
                end_facts = (
                    "implementation session COMPLETED its final report; the "
                    "CLI process lingered and was reaped after a short grace "
                    "— this was a successful session, not a failure"
                )
            elif meta.get("deadline_exceeded") and meta.get("completed_before_kill"):
                end_facts = (
                    "implementation session COMPLETED its final report "
                    "before the deadline kill — the kill reflects a lingering "
                    "process, not unfinished work"
                )
            elif result.success:
                end_facts = "implementation session ended naturally"
            elif meta.get("deadline_exceeded"):
                end_facts = (
                    "implementation session was KILLED BY ITS OWN DEADLINE "
                    f"after {meta.get('elapsed_seconds', 0):.0f}s"
                )
            else:
                end_facts = (
                    f"implementation session died prematurely ({result.error}); "
                    "the deadline was NOT reached — suspect an external or "
                    "self-inflicted kill"
                )
            if meta.get("last_tool"):
                end_facts += f"; last tool call before end: {meta['last_tool']}"
            self._pending_session_end_facts = end_facts

            if not result.success:
                logger.warning(f"[GenericSearch] Implementation failed: {result.error}")
                agent_output = (
                    f"Implementation failed: {result.error}\n\n{agent_output}"
                )
        finally:
            agent.cleanup()
        telemetry = {
            "cost_usd": phase_cost,
            "duration_seconds": time.monotonic() - phase_started,
        }

        # 7. Update RepoMemory for this experiment branch
        run_result_payload = {
            "score": 0,
            "run_had_error": False,
            "error_message": "",
            "error_details": "",
            "feedbacks": "",
        }

        # Extract sections consulted from changes.log
        sections_consulted = []
        try:
            changes_log_path = os.path.join(session.session_folder, "changes.log")
            if os.path.exists(changes_log_path):
                with open(
                    changes_log_path, "r", encoding="utf-8", errors="replace"
                ) as f:
                    sections_consulted = extract_repo_memory_sections_consulted(
                        f.read()
                    )
        except Exception:
            sections_consulted = []
        run_result_payload["repo_memory_sections_consulted"] = sections_consulted

        # Schedule RepoMemory update for session close
        session.schedule_repo_memory_update(
            solution_spec=solution,
            run_result=run_result_payload,
        )

        # 8. Registered-evaluation teardown guard: wait for a live grader
        # and stash any durable-archive recovery BEFORE finalize's rmtree.
        self._recovered_manifest_line = self._await_registered_evaluation(agent_output)

        # 9. Finalize session (commits changes)
        self.workspace.finalize_session(session)

        return agent_output, telemetry

    def _build_implementation_prompt(
        self,
        solution: str,
        problem: str,
        branch_name: str,
        repo_memory_brief: str,
        repo_memory_detail_access_instructions: str,
        previous_errors: str,
    ) -> str:
        """Build the implementation prompt for Claude Code."""
        template = load_prompt(
            "execution/search_strategies/generic/prompts/implementation_claude_code.md"
        )
        return render_prompt(
            template,
            {
                "solution": solution or "(No solution provided)",
                "problem": problem or "(No problem description provided)",
                "branch_name": branch_name,
                "repo_memory_brief": repo_memory_brief or "(No repo memory available)",
                "repo_memory_detail_access_instructions": repo_memory_detail_access_instructions,
                "previous_errors": previous_errors or "(No previous errors)",
                "budget_status": self._render_budget_status(),
                "evaluation_instructions": self._evaluation_instructions(),
            },
        )

    def _manifest_score_of_record(self, node: SearchNode) -> Optional[float]:
        """The granted-class score from the session's last manifest line.

        Registered mode only: the wrapper contractually prints one
        machine-readable KAPSO_EVAL_MANIFEST line per run, so an LLM never
        has to be the parser of record (two live nodes lost real
        measurements to a killed feedback call). The line is model
        output: a present-but-malformed manifest raises. A well-formed
        line for a different class — the agent ran a custom fraction or
        the wrong fidelity — is not this node's canonical measurement and
        returns None (documented default).
        """
        if not self.registered_evaluation_command:
            return None
        output = node.evaluation_output or ""
        last_line = None
        for line in output.splitlines():
            if line.strip().startswith(MANIFEST_MARKER):
                last_line = line.strip()
        if last_line is None:
            return None
        manifest = parse_manifest_line(last_line)
        decision = self.fidelity_decision
        granted_fidelity = decision.eval_fidelity if decision is not None else "full"
        granted_fraction = decision.eval_fraction if decision is not None else 1.0
        if (
            manifest["fidelity"] != granted_fidelity
            or abs(float(manifest["fraction"]) - granted_fraction) > 1e-9
            or int(manifest["seed"]) != self.registered_subsample_seed
        ):
            print(
                "[GenericSearch] Manifest class mismatch: granted "
                f"{granted_fidelity}/{granted_fraction}/"
                f"{self.registered_subsample_seed}, session ran "
                f"{manifest['fidelity']}/{manifest['fraction']}/"
                f"{manifest['seed']} — no mechanical score of record"
            )
            return None
        if "score" not in manifest:
            return None
        return float(manifest["score"])

    def _record_evaluation_attempt(self, node: SearchNode) -> None:
        """Append the node's measurement under the registered evaluator.

        Only the registered wrapper's manifest score becomes an attempt. The
        implementation XML and feedback judge are never score authorities.
        """
        if (
            not self.registered_evaluator_id
            or node.had_error
            or not node.evaluation_valid
        ):
            return
        if not self.registered_evaluation_command:
            raise ValueError("registered evaluator has no evaluation command")
        manifest_score = self._manifest_score_of_record(node)
        if manifest_score is None:
            node.score = None
            node.should_stop = False
            return
        node.score = manifest_score
        decision = self.fidelity_decision
        fraction = decision.eval_fraction if decision is not None else 1.0
        commit_sha = self.workspace.repo.commit(node.branch_name).hexsha
        node.evaluation_attempts.append(
            EvaluationAttempt(
                commit_sha=commit_sha,
                evaluator_id=self.registered_evaluator_id,
                fidelity=node.eval_fidelity,
                fraction=fraction,
                seed=self.registered_subsample_seed,
                score=manifest_score,
                duration_seconds=node.phase_telemetry.get("implementation", {}).get(
                    "duration_seconds"
                ),
            )
        )

    def _execute_registered_evaluation(
        self,
        target: SearchNode,
        *,
        fidelity: str,
        fraction: float,
        deadline_seconds: Optional[float],
    ) -> Optional[float]:
        """Frame-run the registered evaluation on an existing artifact.

        This is the staged-execution-ownership step from the design: the
        eval-only runs whose integrity matters most execute under Kapso's
        own deadline-bounded subprocess, not inside an agent session. The
        deadline is the affordability window and an overrun is an
        operational outcome, never a campaign failure: the process group
        is killed and the attempt reports None, exactly like a non-zero
        exit. Timing estimates gate admission; they do not kill campaigns.
        """
        command = shlex.split(
            evaluation_command(
                fidelity=fidelity,
                fraction=fraction,
                seed=self.registered_subsample_seed,
            )
        )
        run_started = time.monotonic()
        with self.workspace.materialize_ref(target.branch_name) as worktree:
            # The branch's own evaluation tree is whatever version its
            # session ran under — a frame run trusting it would execute a
            # RETIRED evaluator while labeling the attempt with the head's
            # id (observed live: a bridge labeled v2 executed the branch's
            # v1 tree). The registered head is the only ruler frame runs
            # execute.
            self._sync_registered_evaluation(worktree)
            if self.registered_data_manifest:
                data_problem = verify_data_manifest(
                    worktree, self.registered_data_manifest
                )
                if data_problem:
                    print(
                        "[GenericSearch] Registered evaluation refused: "
                        f"{data_problem}"
                    )
                    return None
            # The frame emits a handful of lines plus the manifest — far
            # below pipe capacity — so draining once at exit cannot
            # deadlock the child.
            process = subprocess.Popen(
                command,
                cwd=worktree,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            while process.poll() is None:
                overran = (
                    deadline_seconds is not None
                    and time.monotonic() - run_started >= deadline_seconds
                )
                if overran:
                    os.killpg(process.pid, signal.SIGTERM)
                    grace = time.monotonic() + _FRAME_RUN_KILL_GRACE_SECONDS
                    while process.poll() is None and time.monotonic() < grace:
                        time.sleep(0.2)
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    print(
                        "[GenericSearch] Registered evaluation exceeded its "
                        f"{deadline_seconds:.0f}s affordability window; "
                        "recorded as a failed attempt"
                    )
                    return None
                time.sleep(0.5)
            stdout, stderr = process.communicate()
        duration = time.monotonic() - run_started
        if process.returncode != 0:
            print(
                "[GenericSearch] Registered evaluation failed "
                f"(exit {process.returncode}): {stderr}"
            )
            return None
        manifest = parse_manifest_line(stdout)
        score = float(manifest["score"])
        target.evaluation_attempts.append(
            EvaluationAttempt(
                commit_sha=self.workspace.repo.commit(target.branch_name).hexsha,
                evaluator_id=self.registered_evaluator_id,
                fidelity=fidelity,
                fraction=fraction,
                seed=self.registered_subsample_seed,
                score=score,
                duration_seconds=duration,
            )
        )
        if self.record_eval_duration is not None:
            # Feed the measured duration back into the timing model: real
            # full-scale runs replace calibration extrapolation (samples
            # persist in the registry; the provider-backed policy sees the
            # tightened upper immediately).
            self.record_eval_duration(fraction=fraction, duration_seconds=duration)
        return score

    def _run_validate(self, decision: FidelityDecision) -> SearchNode:
        """Execute a VALIDATE grant: one full measurement of the target."""
        target = self.node_history[decision.target_node_id]
        print(
            f"[GenericSearch] VALIDATE: full evaluation of node "
            f"{target.node_id} ({target.branch_name})"
        )
        score = self._execute_registered_evaluation(
            target,
            fidelity="full",
            fraction=1.0,
            deadline_seconds=decision.deadline_seconds,
        )
        if score is not None:
            print(
                f"[GenericSearch] VALIDATE complete: node {target.node_id} "
                f"full score {score}"
            )
        return target

    def run_bridge_evaluation(
        self,
        node: SearchNode,
        *,
        fidelity: str,
        fraction: float,
        deadline_seconds: Optional[float],
    ) -> bool:
        """Re-measure one artifact under the new evaluator head.

        The artifact-gone fallback is mechanical: a branch that no longer
        resolves cannot bridge, and the caller falls to the next candidate.
        """
        branch_names = {head.name for head in self.workspace.repo.heads}
        if node.branch_name not in branch_names:
            print(
                f"[GenericSearch] Bridge skipped: branch "
                f"{node.branch_name!r} no longer exists"
            )
            return False
        score = self._execute_registered_evaluation(
            node,
            fidelity=fidelity,
            fraction=fraction,
            deadline_seconds=deadline_seconds,
        )
        if score is None:
            return False
        # A successful bridge is a fresh, frame-run measurement: it
        # supersedes an evaluation_valid=False verdict that described the
        # OLD (defective-evaluator) measurement. Without this the live
        # requester stayed invalid forever — excluded from parenting and
        # delivery despite carrying an honest new-head score. Tampering
        # nodes never reach the bridge (integrity errors are filtered).
        node.evaluation_valid = True
        return True

    def refresh_score_projections(self, comparability: ComparabilityClass) -> None:
        """Re-project every node's score under one canonical ruler.

        The selectors stay dumb: after an evaluator transition, nodes never
        measured under the new ruler project None — and None never wins.
        """
        for node in self.node_history:
            node.score = project_score(node, comparability)

    def _sync_registered_evaluation(self, session_folder: str) -> None:
        """Overwrite the session's evaluation tree with the registered one."""
        source = os.path.join(self.workspace_dir, "kapso_evaluation")
        destination = os.path.join(session_folder, "kapso_evaluation")
        shutil.rmtree(destination, ignore_errors=True)
        shutil.copytree(source, destination)

    def _evaluation_instructions(self) -> str:
        """Registered-evaluation contract when a maintainer owns evaluation;
        the historical build-your-own instructions otherwise."""
        if not self.registered_evaluation_command:
            return DEFAULT_EVALUATION_INSTRUCTIONS
        return f"""The evaluation is maintained by the system and is read-and-execute only.

1. **Run the registered evaluation**: `{self.registered_evaluation_command}`
   and capture its full output, including the KAPSO_EVAL_MANIFEST line.
2. **Run it in the FOREGROUND and stay alive until it finishes.** Your
   session exists only while you are actively working: the moment you stop
   responding, the session ends and every process it started is killed. No
   background job survives you, and no completion notification can ever
   reach you — there is no later. Never launch the registered evaluation
   with `&`, `nohup`, or a background task. Full-fidelity builds taking
   many minutes is normal and expected: run the command blocking with a
   generous tool timeout, and if a single call hits its cap, keep
   re-issuing blocking foreground waits until KAPSO_EVAL_MANIFEST is in
   your transcript. Only then write your final response. An evaluation you
   background and abandon scores nothing — the entire iteration is wasted.
3. **Never alter evaluation behavior — at rest or at runtime.** Editing
   anything under `kapso_evaluation/`, rewriting protected data inputs,
   monkey-patching or hooking evaluation modules from your own code
   (e.g. via imports, `sys.modules`, or wrappers), or otherwise
   circumventing any evaluation check all count as tampering: the score
   is voided and the experiment loses. There is no sanctioned bypass.
4. **If you believe the evaluation itself is broken**, do not fix it,
   patch it, or route around it. File a request by including this tag in
   your final response:
   <evaluation_change_request>concrete description of the defect, with the
   exact error output as evidence</evaluation_change_request>
   Then still report your results from the run you attempted. The
   maintainer investigates immediately; a confirmed defect is fixed and
   your work is re-measured first under the corrected evaluation.
5. **Retry on transient crashes** of your own code (max 3 attempts)."""

    def _ensure_technical_difficulties(self, node) -> None:
        """Run the fallback reconstruction when the implementor's
        technical_difficulties tag is missing (crashed or deadline-killed
        session, or simply omitted)."""
        if (node.technical_difficulties or "").strip():
            return
        print(
            "[GenericSearch] technical_difficulties missing — "
            "running fallback reconstruction"
        )
        node.technical_difficulties = generate_technical_difficulties(
            model=self.implementation_model,
            claude_auth_settings=self._claude_auth_settings,
            aws_region=self.aws_region,
            env_strip=self.env_strip,
            effort=self.session_effort,
            timeout_seconds=self._clamped_timeout(self.implementation_timeout),
            workspace_dir=node.workspace_dir or self.workspace_dir,
            solution=node.solution,
            stream_artifact_path=self._session_stream_path(node.branch_name),
        )

    def _await_registered_evaluation(self, output_text: str):
        """Teardown guard for the registered evaluation (relbench finding 14 /
        Issue 2). MUST run BEFORE finalize_session: its rmtree destroys a
        still-running grader's working tree. If the session ended without a
        manifest in its output while the registered evaluation process is
        alive, wait for it (bounded by the live budget clamp). Then attempt
        recovery from the durable run archive — the grader archives the run
        (including manifest.txt) OUTSIDE the workspace before printing the
        manifest line — and return the recovered manifest line (or None).
        """
        if not self.registered_evaluation_command:
            return None
        if MANIFEST_MARKER in (output_text or ""):
            return None

        # A distinctive fragment of the registered command for /proc matching:
        # prefer the script path token; fall back to the full command string.
        tokens = [t for t in self.registered_evaluation_command.split() if ".py" in t]
        needle = tokens[0] if tokens else self.registered_evaluation_command

        def _live_eval_pid():
            for pid in os.listdir("/proc"):
                if not pid.isdigit():
                    continue
                cmdline_path = os.path.join("/proc", pid, "cmdline")
                if not os.path.exists(cmdline_path):
                    continue
                with open(cmdline_path, "rb") as fh:
                    cmdline = fh.read().replace(b"\0", b" ").decode("utf-8", "replace")
                if needle in cmdline:
                    return int(pid)
            return None

        bound = self._clamped_timeout(self.implementation_timeout)
        waited = 0.0
        pid = _live_eval_pid()
        if pid is not None:
            print(
                f"[GenericSearch] Registered evaluation still running "
                f"(pid {pid}) after session end — waiting up to {bound:.0f}s "
                "before teardown"
            )
        while pid is not None and waited < bound:
            time.sleep(5)
            waited += 5
            pid = _live_eval_pid()

        if not self.registered_evaluation_archive_glob:
            return None
        started = getattr(self, "_last_session_started_ts", 0.0)
        candidates = []
        for runs_root in glob.glob(self.registered_evaluation_archive_glob):
            for entry in glob.glob(os.path.join(runs_root, "run_*")):
                if os.path.isdir(entry) and os.path.getmtime(entry) > started:
                    candidates.append(entry)
        for run_dir in sorted(candidates, key=os.path.getmtime, reverse=True):
            manifest_path = os.path.join(run_dir, "manifest.txt")
            if not os.path.isfile(manifest_path):
                continue
            with open(manifest_path, "r", encoding="utf-8") as fh:
                line = fh.read().strip()
            if not line.startswith(MANIFEST_MARKER):
                continue
            print(
                "[GenericSearch] Recovered registered-evaluation manifest "
                f"from durable archive: {run_dir}"
            )
            return line
        return None

    def _session_stream_path(self, branch_name: str) -> str:
        """Per-session stream artifact location (survives session kills)."""
        stream_dir = os.path.join(self.workspace_dir, ".kapso", "sessions", branch_name)
        os.makedirs(stream_dir, exist_ok=True)
        return os.path.join(stream_dir, "stream.jsonl")

    def _clamped_timeout(self, configured_seconds: float) -> float:
        """Bound an agent deadline by the searchable budget, when known.

        The snapshot is frozen at iteration start; the monotonic anchor
        discounts whatever this iteration's earlier phases already burned,
        so implementation clamps against what actually remains after
        ideation, not the iteration-start remainder.
        """
        if self.budget_snapshot is None:
            return configured_seconds
        drift = (
            time.monotonic() - self.budget_snapshot_monotonic
            if self.budget_snapshot_monotonic is not None
            else 0.0
        )
        return self.budget_snapshot.clamp_timeout(
            configured_seconds, elapsed_since_snapshot=drift
        )

    def _render_budget_status(self) -> str:
        """Deterministic budget block for prompts. Advisory only — never a
        protection mechanism; enforcement is the deadline clamp and the
        orchestrator's gates."""
        snapshot = self.budget_snapshot
        if snapshot is None:
            return (
                f"Iteration {self.iteration_count} — no budget information "
                "available."
            )
        position = (
            f"Iteration {snapshot.iteration_index + 1} of "
            f"{snapshot.max_iterations}."
        )
        if snapshot.time_budget_seconds is None and snapshot.cost_budget_usd is None:
            return f"{position} No time or cost budget is set."
        parts = [position]
        if snapshot.time_budget_seconds is not None:
            parts.append(
                f"Elapsed {snapshot.elapsed_seconds / 60:.0f} of "
                f"{snapshot.time_budget_seconds / 60:.0f} budgeted minutes."
            )
            if snapshot.finalization_reserve_seconds > 0:
                searchable = max(snapshot.remaining_after_reserve, 0.0)
                parts.append(
                    "Finalization reserve escrowed: "
                    f"{snapshot.finalization_reserve_seconds / 60:.0f} "
                    "minutes; searchable time remaining: "
                    f"{searchable / 60:.0f} minutes."
                )
        if snapshot.cost_budget_usd is not None:
            parts.append(
                f"Spent ${snapshot.cost_usd:.2f} of "
                f"${snapshot.cost_budget_usd:.2f}."
            )
        return " ".join(parts)

    def get_experiment_history(self, best_last: bool = False) -> List[SearchNode]:
        """Return all nodes, optionally sorted by score (unscored sort worst)."""
        if best_last:
            return sorted(
                self.node_history,
                key=lambda node: (
                    not node.had_error
                    and node.evaluation_valid
                    and node.score is not None,
                    (
                        0.0
                        if node.score is None
                        else (
                            node.score
                            if self.problem_handler.maximize_scoring
                            else -node.score
                        )
                    ),
                ),
            )
        return self.node_history

    def record_finalized_idea_outcome(self, node: SearchNode) -> None:
        """Persist the idea result after orchestrator-side evaluation is final."""
        if node.idea_id is None or node.selection_batch_id is None:
            raise ValueError("finalized generic node has no idea provenance")
        archive = self._ensure_idea_archive()
        ideas = {idea.idea_id: idea for idea in archive.state.ideas}
        if node.idea_id not in ideas:
            raise ValueError("finalized generic node references an unknown idea")
        outcome = build_idea_outcome(
            node=node,
            idea=ideas[node.idea_id],
            nodes_by_id={item.node_id: item for item in self.node_history},
            objective_direction=(
                ObjectiveDirection.MAXIMIZE
                if self.problem_handler.maximize_scoring
                else ObjectiveDirection.MINIMIZE
            ),
        )
        if outcome is None:
            return
        evidence_writeback = build_evaluator_evidence_writeback(
            node.external_evaluation_metadata,
            idea=ideas[node.idea_id],
            archive_state=archive.state,
            observed_at=node.started_at,
        )
        outcome = evidence_writeback.apply_to_outcome(outcome)
        archive.record_outcome(
            node.idea_id,
            outcome,
            claim_updates=evidence_writeback.claim_updates,
            gap_updates=evidence_writeback.gap_updates,
            expected_revision=archive.revision,
        )
        self.active_batch_id = self._archive_active_batch_id(archive)

    def reconcile_experiment_memory(
        self,
        store: ExperimentHistoryStore,
    ) -> None:
        """Reconcile the archive, executed memory, and checkpoint node prefix."""
        archive = self._ensure_idea_archive()
        objective_direction = (
            "maximize" if self.problem_handler.maximize_scoring else "minimize"
        )
        if store.objective_direction != objective_direction:
            raise ValueError("experiment memory objective conflicts with strategy")
        if store.require_idea_links is not True:
            raise ValueError("generic experiment memory must require idea links")

        checkpoint_node_count = len(self.node_history)
        if len(store.experiments) < checkpoint_node_count:
            for node in self.node_history[len(store.experiments) :]:
                store.add_experiment(node)
                self.record_finalized_idea_outcome(node)
        for index, record in enumerate(store.experiments[:checkpoint_node_count]):
            projected = ExperimentRecord.from_node(
                self.node_history[index],
                objective_direction,
                True,
            )
            if projected != record:
                stable_checkpoint_identity = (
                    projected.node_id,
                    projected.idea_id,
                    projected.selection_batch_id,
                    projected.parent_node_id,
                    projected.solution,
                    projected.objective_direction,
                )
                stable_record_identity = (
                    record.node_id,
                    record.idea_id,
                    record.selection_batch_id,
                    record.parent_node_id,
                    record.solution,
                    record.objective_direction,
                )
                checkpoint_node = self.node_history[index]
                if (
                    stable_checkpoint_identity != stable_record_identity
                    or not checkpoint_node.recoverable_error
                    or record.execution_revision <= projected.execution_revision
                ):
                    raise ValueError("checkpoint node conflicts with experiment memory")
                recovered = self._node_from_experiment_record(record, archive)
                if (
                    ExperimentRecord.from_node(
                        recovered,
                        objective_direction,
                        True,
                    )
                    != record
                ):
                    raise ValueError(
                        "newer experiment-memory revision cannot be reconstructed"
                    )
                self.node_history[index] = recovered
                self.record_finalized_idea_outcome(recovered)

        for record in store.experiments[len(self.node_history) :]:
            if record.node_id != len(self.node_history):
                raise ValueError("experiment memory tail is not contiguous")
            node = self._node_from_experiment_record(record, archive)
            round_trip = ExperimentRecord.from_node(
                node,
                objective_direction,
                True,
            )
            if round_trip != record:
                raise ValueError("experiment memory record cannot be reconstructed")
            self.node_history.append(node)
            self.record_finalized_idea_outcome(node)

        for record in store.experiments:
            idea = archive.get_idea(record.idea_id)
            if idea.outcome is None and not record.recoverable_error:
                self.record_finalized_idea_outcome(self.node_history[record.node_id])

        self.active_batch_id = self._archive_active_batch_id(archive)
        if self.active_batch_id is None:
            return
        active = archive.get_batch(self.active_batch_id)
        if active.status != BatchStatus.BRIDGED:
            return
        if active.selection is None:
            raise ValueError("bridged idea batch has no selection")
        idea = archive.get_idea(active.selection.selected_idea_id)
        if idea.experiment_node_id is None:
            raise ValueError("bridged idea has no experiment node")
        if idea.experiment_node_id < len(self.node_history):
            node = self.node_history[idea.experiment_node_id]
            if node.idea_id != idea.idea_id or not node.recoverable_error:
                raise ValueError("unfinished idea conflicts with checkpoint node")
            return
        if idea.experiment_node_id != len(self.node_history):
            raise ValueError("bridged experiment node is not contiguous")
        branch_name = f"generic_exp_{idea.experiment_node_id}"
        heads = {head.name for head in self.workspace.repo.heads}
        if branch_name not in heads:
            self.workspace.repo.create_head(branch_name, idea.resolved_parent.git_ref)
        merge_base = self.workspace.repo.git.merge_base(
            idea.resolved_parent.git_ref,
            branch_name,
        ).strip()
        if merge_base != idea.resolved_parent.git_ref:
            raise ValueError("interrupted experiment branch changed ancestry")
        node = SearchNode(
            node_id=idea.experiment_node_id,
            parent_node_id=idea.resolved_parent.node_id,
            idea_id=idea.idea_id,
            selection_batch_id=active.batch_id,
            solution=idea.proposal,
            branch_name=branch_name,
            parent_branch_name=idea.resolved_parent.branch_name,
            implementation_base_ref=idea.resolved_parent.git_ref,
            diff_base_ref=idea.resolved_parent.diff_base_ref,
            feedback_base_ref=idea.resolved_parent.feedback_base_ref,
            feedback="Implementation was interrupted before durable finalization.",
            score=None,
            evaluation_valid=False,
            started_at=idea.created_at,
            had_error=True,
            recoverable_error=True,
            error_message="interrupted_before_experiment_persistence",
            workspace_dir=self.workspace_dir,
            technical_difficulties=(
                "Resume the interrupted implementation on its original branch."
            ),
        )
        ideation_telemetry = self._ideation_phase_telemetry(
            IdeationEngineTelemetry(
                generation_calls=active.generation_calls,
                selection_call=active.selection_call,
                embedding=active.embedding_telemetry,
            )
        )
        node.phase_telemetry["ideation"] = ideation_telemetry
        node.duration_seconds = ideation_telemetry["duration_seconds"]
        node.cost_usd = ideation_telemetry["cost_usd"]
        self.node_history.append(node)

    def _node_from_experiment_record(
        self,
        record: ExperimentRecord,
        archive: IdeaArchive,
    ) -> SearchNode:
        """Rebuild the strict SearchNode projection from durable execution data."""
        if record.idea_id is None or record.selection_batch_id is None:
            raise ValueError("generic experiment record lacks idea provenance")
        idea = archive.get_idea(record.idea_id)
        if (
            idea.experiment_node_id != record.node_id
            or idea.selected_in_batch_id != record.selection_batch_id
            or idea.proposal != record.solution
            or idea.resolved_parent.node_id != record.parent_node_id
        ):
            raise ValueError("experiment record conflicts with idea archive")
        node = SearchNode(
            node_id=record.node_id,
            parent_node_id=record.parent_node_id,
            execution_revision=record.execution_revision,
            idea_id=record.idea_id,
            selection_batch_id=record.selection_batch_id,
            solution=record.solution,
            branch_name=record.branch_name,
            parent_branch_name=idea.resolved_parent.branch_name,
            implementation_base_ref=idea.resolved_parent.git_ref,
            diff_base_ref=idea.resolved_parent.diff_base_ref,
            feedback_base_ref=idea.resolved_parent.feedback_base_ref,
            feedback=record.feedback,
            score=record.raw_score,
            evaluation_valid=record.evaluation_valid,
            evaluation_provenance=record.evaluation_provenance,
            evaluation_integrity_error=record.evaluation_integrity_error,
            metrics=dict(record.metrics),
            primary_metric=record.primary_metric,
            external_evaluation_metadata=dict(record.external_evaluation_metadata),
            external_evaluation_error=record.external_evaluation_error,
            duration_seconds=record.duration_seconds,
            cost_usd=record.cost_usd,
            started_at=record.timestamp,
            build_fidelity=record.build_fidelity,
            eval_fidelity=record.eval_fidelity,
            evaluation_attempts=list(record.evaluation_attempts),
            phase_telemetry={
                phase: dict(measurements)
                for phase, measurements in record.phase_telemetry.items()
            },
            had_error=record.had_error,
            recoverable_error=record.recoverable_error,
            error_message=record.error_message,
            workspace_dir=self.workspace_dir,
            technical_difficulties=record.technical_difficulties,
        )
        if record.branch_name:
            heads = {head.name for head in self.workspace.repo.heads}
            if record.branch_name not in heads:
                raise ValueError("experiment record references a missing branch")
            node.code_diff = self._get_code_diff(
                record.branch_name,
                idea.resolved_parent.diff_base_ref,
            )
        return node

    def get_best_experiment(self) -> Optional[SearchNode]:
        """Return the best successful SCORED node — a node whose evaluation
        never completed (score=None) can never be best; on minimize metrics
        it would otherwise key as 0 and out-rank every real score."""
        valid = [
            node
            for node in self.node_history
            if not node.had_error and node.evaluation_valid and node.score is not None
        ]
        if not valid:
            return None
        return max(
            valid,
            key=lambda x: (
                x.score if self.problem_handler.maximize_scoring else -x.score
            ),
        )

    def get_deliverable_experiment(self) -> Optional[SearchNode]:
        """The committed-slot winner: evidence tiers, never raw scores.

        Parent selection explores on projected scores (the four-bests
        split); the deliverable follows the tier walk under the registered
        evaluator, so an unvalidated fast leader cannot displace a
        full-tier candidate at delivery. Without registered evidence the
        score leader stands.
        """
        if self.registered_evaluator_id:
            committed = select_committed_candidate(
                self.node_history,
                evaluator_id=self.registered_evaluator_id,
                maximize=self.problem_handler.maximize_scoring,
            )
            if committed is not None:
                return committed
        return self.get_best_experiment()

    def get_deliverable_score(self) -> Optional[float]:
        """The deliverable's authoritative measurement.

        Prefers the full-fidelity class under the registered evaluator —
        the score the campaign actually vouches for — over the canonical
        (possibly fast) projection stored on node.score.
        """
        node = self.get_deliverable_experiment()
        if node is None:
            return None
        if self.registered_evaluator_id:
            full_score = project_score(
                node,
                ComparabilityClass(
                    evaluator_id=self.registered_evaluator_id,
                    fidelity="full",
                    fraction=1.0,
                    seed=self.registered_subsample_seed,
                ),
            )
            if full_score is not None:
                return full_score
        return node.score

    def checkout_to_best_experiment_branch(self) -> Optional[str]:
        """Checkout and return the deliverable node's branch."""
        best = self.get_deliverable_experiment()
        if best:
            print(
                "[GenericSearch] Checking out deliverable branch: "
                f"{best.branch_name} (score={best.score})"
            )
            self.workspace.switch_branch(best.branch_name)
            return best.branch_name
        else:
            print("[GenericSearch] No successful experiments to checkout")
            return None

    # =========================================================================
    # Feedback and Result Extraction (Generic-specific)
    # =========================================================================

    def _generate_feedback(self, node: SearchNode) -> SearchNode:
        """
        Generate feedback for a node using the FeedbackGenerator.

        Updates the node in-place with feedback, score, and should_stop.

        Args:
            node: SearchNode with solution, evaluation_output, code_changes_summary populated

        Returns:
            The same node with feedback, score, should_stop populated
        """
        if self.feedback_generator is None:
            print("[GenericSearch] No feedback generator configured, skipping feedback")
            return node

        if not self.goal:
            print("[GenericSearch] Warning: No goal set, skipping feedback generation")
            return node

        print(f"[GenericSearch] Generating feedback for node {node.node_id}...")

        try:
            feedback_result = self.feedback_generator.generate(
                goal=self.goal,
                idea=node.solution,
                code_changes_summary=node.code_changes_summary,
                base_branch=node.feedback_base_ref or node.parent_branch_name,
                head_branch=node.branch_name,
                evaluation_script_path=node.evaluation_script_path,
                evaluation_result=node.evaluation_output,
                workspace_dir=node.workspace_dir,
                session_end_facts=getattr(self, "_pending_session_end_facts", ""),
                timeout_seconds=self._clamped_timeout(
                    self.feedback_generator.configured_timeout_seconds
                ),
            )

            # Update node with feedback results
            node.feedback = feedback_result.feedback
            node.evaluation_valid = feedback_result.evaluation_valid
            node.score = (
                feedback_result.score if feedback_result.evaluation_valid else None
            )
            # In registered mode the manifest line is the score of record;
            # the judge's extraction is a cross-check, and the judge keeps
            # its validity power (an invalid evaluation stays scoreless).
            manifest_score = self._manifest_score_of_record(node)
            if manifest_score is not None and node.evaluation_valid:
                if node.score is not None and abs(node.score - manifest_score) > 1e-6:
                    print(
                        "[GenericSearch] Score cross-check: feedback "
                        f"extracted {node.score}, the manifest says "
                        f"{manifest_score}; the manifest is the score "
                        "of record"
                    )
                node.score = manifest_score
            node.should_stop = feedback_result.stop and feedback_result.evaluation_valid
            if feedback_result.duration_seconds is not None:
                self._accumulate_phase_telemetry(
                    node,
                    "feedback",
                    {
                        "cost_usd": feedback_result.cost_usd,
                        "duration_seconds": feedback_result.duration_seconds,
                    },
                )

            print(
                f"[GenericSearch] Feedback generated: stop={node.should_stop}, score={node.score}"
            )

        except Exception as e:
            print(f"[GenericSearch] Error generating feedback: {e}")
            node.feedback = f"Error generating feedback: {e}"
            node.should_stop = False

        return node

    def _extract_agent_result(self, agent_output: str) -> dict:
        """
        Extract structured result from agent output using XML tags.

        The agent is instructed to return results in XML tags:
        <code_changes_summary>...</code_changes_summary>
        <evaluation_script_path>...</evaluation_script_path>
        <evaluation_output>...</evaluation_output>
        <score>...</score>
        <technical_difficulties>...</technical_difficulties>

        Args:
            agent_output: Raw output from the developer agent

        Returns:
            Complete parsed implementation result.

        Raises:
            ValueError: The successful-session output violates the XML contract.
        """
        if not isinstance(agent_output, str) or not agent_output.strip():
            raise ValueError("implementation output must be non-empty")
        result = {}
        tags = (
            "code_changes_summary",
            "evaluation_script_path",
            "evaluation_output",
            "score",
            "technical_difficulties",
        )

        for tag in tags:
            pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
            matches = re.findall(pattern, agent_output, re.DOTALL)
            if len(matches) != 1:
                raise ValueError(
                    f"implementation output requires exactly one <{tag}> tag"
                )
            value = matches[0].strip()
            if tag == "score":
                if value.lower() == "null":
                    result[tag] = None
                else:
                    try:
                        result[tag] = float(value)
                    except ValueError as exc:
                        raise ValueError(
                            "implementation score must be finite or null"
                        ) from exc
                    if not math.isfinite(result[tag]):
                        raise ValueError("implementation score must be finite or null")
            else:
                if not value:
                    raise ValueError(f"implementation output <{tag}> must be non-empty")
                result[tag] = value
        return result

    # =========================================================================
    # Checkpoint Methods
    # =========================================================================

    def dump_state(self) -> Dict[str, Any]:
        """Return the exact v3 generic-search checkpoint projection."""
        archive = self._ensure_idea_archive()
        active_batch_id = self._archive_active_batch_id(archive)
        self.active_batch_id = active_batch_id
        return {
            "schema": GENERIC_SEARCH_STATE_SCHEMA,
            "campaign_id": self.ideation_campaign_id,
            "idea_archive_schema": IDEA_ARCHIVE_SCHEMA,
            "archive_revision": archive.revision,
            "active_batch_id": active_batch_id,
            "node_history": [node.to_dict() for node in self.node_history],
            "iteration_count": self.iteration_count,
            "previous_errors": list(self.previous_errors),
            "evaluation_integrity": (self.dump_evaluation_integrity_state()),
            "scores_evaluator_id": self.scores_evaluator_id,
            "evaluator_transition": self.evaluator_transition,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Restore only the exact v3 state and reconcile archive advancement."""
        if not isinstance(state, dict) or set(state) != _GENERIC_SEARCH_STATE_FIELDS:
            raise ValueError("GenericSearch checkpoint fields are incompatible")
        if state["schema"] != GENERIC_SEARCH_STATE_SCHEMA:
            raise ValueError("GenericSearch checkpoint schema is incompatible")
        if state["idea_archive_schema"] != IDEA_ARCHIVE_SCHEMA:
            raise ValueError("GenericSearch idea archive schema is incompatible")
        campaign_id = state["campaign_id"]
        if not isinstance(campaign_id, str) or not campaign_id.startswith("campaign_"):
            raise ValueError("GenericSearch checkpoint campaign id is invalid")
        saved_revision = state["archive_revision"]
        if (
            isinstance(saved_revision, bool)
            or not isinstance(saved_revision, int)
            or saved_revision < 0
        ):
            raise ValueError("GenericSearch checkpoint archive revision is invalid")
        saved_active_batch_id = state["active_batch_id"]
        if saved_active_batch_id is not None and (
            not isinstance(saved_active_batch_id, str)
            or not saved_active_batch_id.startswith("batch_")
        ):
            raise ValueError("GenericSearch checkpoint active batch id is invalid")

        archive_path = self._workspace_config_path(self.ideation_config["archive_path"])
        if not archive_path.is_file():
            raise ValueError("GenericSearch checkpoint idea archive is missing")
        self.ideation_campaign_id = campaign_id
        self.idea_archive = IdeaArchive(archive_path, campaign_id)
        archive = self._ensure_idea_archive()
        if archive.revision < saved_revision:
            raise ValueError("idea archive revision is behind the run checkpoint")
        archive_active_batch_id = self._archive_active_batch_id(archive)
        if archive.revision == saved_revision:
            if archive_active_batch_id != saved_active_batch_id:
                raise ValueError("checkpoint active batch conflicts with idea archive")
        elif saved_active_batch_id not in {None, archive_active_batch_id}:
            saved_batch = archive.get_batch(saved_active_batch_id)
            if saved_batch.status not in {
                BatchStatus.COMPLETED,
                BatchStatus.ABANDONED,
            }:
                raise ValueError("archive advancement changed the active batch")
        self.active_batch_id = archive_active_batch_id

        raw_history = state["node_history"]
        if not isinstance(raw_history, list):
            raise ValueError("GenericSearch checkpoint node_history must be a list")
        node_fields = {item.name for item in fields(SearchNode)}
        if any(
            not isinstance(node_data, dict) or set(node_data) != node_fields
            for node_data in raw_history
        ):
            raise ValueError("GenericSearch checkpoint node fields are incompatible")
        self.node_history = [
            SearchNode.from_dict(node_data) for node_data in raw_history
        ]
        node_ids = [node.node_id for node in self.node_history]
        if node_ids != list(range(len(self.node_history))):
            raise ValueError(
                "GenericSearch checkpoint node IDs must be unique, ordered, "
                "and contiguous from zero"
            )

        iteration_count = state["iteration_count"]
        if (
            isinstance(iteration_count, bool)
            or not isinstance(iteration_count, int)
            or iteration_count < 0
        ):
            raise ValueError(
                "GenericSearch checkpoint iteration_count must be non-negative"
            )
        if iteration_count < len(self.node_history):
            raise ValueError(
                "GenericSearch checkpoint iteration_count cannot be smaller "
                "than node_history: every node consumed an iteration"
            )
        self.iteration_count = iteration_count

        previous_errors = state["previous_errors"]
        if not isinstance(previous_errors, list) or not all(
            isinstance(error, str) for error in previous_errors
        ):
            raise ValueError("GenericSearch checkpoint previous_errors must be strings")
        self.previous_errors = list(previous_errors)

        nodes_by_id = {node.node_id: node for node in self.node_history}
        ideas_by_id = {idea.idea_id: idea for idea in archive.state.ideas}
        batches_by_id = {batch.batch_id: batch for batch in archive.state.batches}
        for node in self.node_history:
            if node.idea_id is None or node.selection_batch_id is None:
                raise ValueError("GenericSearch checkpoint node lacks idea provenance")
            idea = ideas_by_id.get(node.idea_id)
            batch = batches_by_id.get(node.selection_batch_id)
            if (
                idea is None
                or batch is None
                or idea.selected_in_batch_id != node.selection_batch_id
                or idea.experiment_node_id != node.node_id
                or batch.selection is None
                or batch.selection.selected_idea_id != node.idea_id
                or node.solution != idea.proposal
                or node.parent_node_id != idea.resolved_parent.node_id
            ):
                raise ValueError("GenericSearch checkpoint idea linkage is corrupt")
            if node.parent_node_id is None:
                if node.parent_branch_name not in {"", "main"}:
                    raise ValueError(
                        "GenericSearch checkpoint baseline parent branch "
                        "must be main"
                    )
                continue
            parent = nodes_by_id.get(node.parent_node_id)
            if parent is None or parent.node_id >= node.node_id:
                raise ValueError(
                    "GenericSearch checkpoint parent_node_id must reference "
                    "an earlier node"
                )
            if (
                node.parent_branch_name
                and node.parent_branch_name != parent.branch_name
            ):
                raise ValueError(
                    "GenericSearch checkpoint parent node and branch do not " "match"
                )
        linked_node_ids = sorted(
            idea.experiment_node_id
            for idea in archive.state.ideas
            if idea.experiment_node_id is not None
        )
        if linked_node_ids != list(range(len(linked_node_ids))):
            raise ValueError("idea archive experiment node links are not contiguous")
        if node_ids != linked_node_ids[: len(node_ids)]:
            raise ValueError("checkpoint nodes are not an archive-linked prefix")
        self.load_evaluation_integrity_state(state["evaluation_integrity"])

        scores_evaluator_id = state["scores_evaluator_id"]
        if not isinstance(scores_evaluator_id, str):
            raise ValueError(
                "GenericSearch checkpoint scores_evaluator_id must be a " "string"
            )
        self.scores_evaluator_id = scores_evaluator_id
        transition = state["evaluator_transition"]
        if transition is not None and (
            not isinstance(transition, dict)
            or transition.get("status") not in {"pending", "anchored"}
            or not isinstance(transition.get("old_evaluator_id"), str)
            or not isinstance(transition.get("new_evaluator_id"), str)
            or (
                "priority_node_id" in transition
                and (
                    isinstance(transition["priority_node_id"], bool)
                    or not isinstance(transition["priority_node_id"], int)
                )
            )
        ):
            raise ValueError("GenericSearch checkpoint evaluator_transition is invalid")
        self.evaluator_transition = transition

    @staticmethod
    def _archive_active_batch_id(archive: IdeaArchive) -> Optional[str]:
        active = tuple(
            batch
            for batch in archive.state.batches
            if batch.status not in {BatchStatus.COMPLETED, BatchStatus.ABANDONED}
        )
        if len(active) > 1:
            raise ValueError("idea archive contains multiple active batches")
        if not active:
            return None
        newest_iteration = max(batch.iteration_index for batch in archive.state.batches)
        if active[0].iteration_index != newest_iteration:
            raise ValueError("active idea batch is not the newest batch")
        return active[0].batch_id
