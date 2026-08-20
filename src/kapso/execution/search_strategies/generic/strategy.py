# Generic Search Strategy
#
# The main search strategy for general problem solving.
# Simple sequential search: generate one solution per iteration,
# implement it, and keep track of the best result.
#
# Key features:
# - Uses Claude Code as the ideation agent with MCP gates
# - Connected to MCP gates (idea, code, research, experiment_history, repo_memory) for external knowledge
# - Read-only access to codebase during ideation
# - Full RepoMemory access via MCP tools

import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from kapso.execution.search_strategies.base import (
    SearchStrategy,
    SearchStrategyConfig,
    SearchNode,
)
from kapso.execution.search_strategies.factory import register_strategy
from kapso.execution.fidelity import (
    FULL_PASSTHROUGH,
    PROFILE_VALIDATE,
    ComparabilityClass,
    FidelityDecision,
    project_score,
    select_committed_candidate,
)
from concurrent.futures import ThreadPoolExecutor
from kapso.execution.search_strategies.generic import codex_ideation
from kapso.execution.search_strategies.generic.expansion_lanes import (
    lane_env_overlay,
    normalize_node_expansion,
    pick_representative,
    render_lane_brief,
    validate_node_expansion_config,
)
from kapso.execution.search_strategies.generic.feedback_flow import (
    extract_agent_result,
    generate_feedback,
)
from kapso.execution.search_strategies.generic.implementation import (
    build_implementation_prompt,
    ensure_technical_difficulties,
    run_implementation,
)
from kapso.execution.search_strategies.generic.lens_planning import (
    design_axes_brief,
    normalize_design_axes,
    normalize_ideation_lens_planner,
    resolve_member_lenses,
    run_lens_planner_session,
    validate_lens_planner_against_ensemble,
)
from kapso.execution.search_strategies.generic.registered_evaluation import (
    await_registered_evaluation,
    evaluation_instructions,
    execute_registered_evaluation,
    manifest_of_record,
    manifest_score_of_record,
    record_evaluation_attempt,
    sync_registered_evaluation,
)
from kapso.execution.search_strategies.generic.shared_cache import (
    SHARED_CACHE_ENV_VAR,
    build_shared_artifacts_brief,
)
from kapso.execution.memories.repo_memory import RepoMemoryManager
from kapso.core.prompt_loader import load_prompt, render_prompt


logger = logging.getLogger(__name__)

PARENT_POLICIES = frozenset({"best", "baseline"})

# A deadline-killed ideation whose streamed text is shorter than this holds
# no consumable plan; the explicit fallback is more honest than salvage.
MIN_IDEATION_SALVAGE_CHARS = 200

# Ensemble ideation: members run in parallel, so the member share is
# wall-clock for the whole fan-out; the selector gets the remainder with a
# floor below which a read-verify-choose session cannot do useful work.
ENSEMBLE_MEMBER_TIME_FRACTION = 0.7
ENSEMBLE_SELECTOR_TIME_FRACTION = 0.3
ENSEMBLE_SELECTOR_MIN_SECONDS = 240
# Default only — the live value is search_strategy.params
# ideation_candidates_per_member (see GenericSearch.__init__). A wider pool
# gives the selector more to choose from and keeps K-way expansion alive
# when one member under-delivers.
ENSEMBLE_CANDIDATES_PER_MEMBER = 3

# Extraction artifacts (prompt echoes, stream duplicates) are shorter than
# any real plan; drop them before the selector sees the pool.
MIN_ENSEMBLE_CANDIDATE_CHARS = 80

# A candidate that is all headers and [placeholders] is a format skeleton,
# not a plan — require this much real content after stripping them.
MIN_ENSEMBLE_CANDIDATE_CONTENT_CHARS = 40


def is_degenerate_ensemble_candidate(text: str) -> bool:
    """True for skeleton/echo artifacts that must never reach the selector."""
    stripped = text.strip()
    if len(stripped) < MIN_ENSEMBLE_CANDIDATE_CHARS:
        return True
    content = re.sub(r"^\s*#.*$", "", stripped, flags=re.MULTILINE)
    content = re.sub(r"\[[^\]]*\]", "", content)
    content = re.sub(r"\s+", "", content)
    return len(content) < MIN_ENSEMBLE_CANDIDATE_CONTENT_CHARS

DEFAULT_MEMBER_LENS = "no specific lens — judge freely"


# A selected solution is a full implementation spec (2,000-4,400 chars across
# runs). Contest 5 (2026-08-06) accepted 3-char bodies ("and") as solutions
# 1-2, so the empty-pool fallback never fired and the round ran 2 lanes
# instead of 8 — six lanes of search lost in silence. Anything under this
# floor is a malformed emission, not a plan.
MIN_SELECTED_SOLUTION_CHARS = 200


def parse_selected_solutions(output: str, expansion_count: int) -> List[str]:
    """Extract the selector's ranked solutions.

    K=1 keeps today's single <solution> contract. K>1 reads <solution_N>
    tags in rank order. A slot that is missing, empty, or shorter than
    MIN_SELECTED_SOLUTION_CHARS is rejected loudly — the caller retries the
    selector and tops up from the candidate pool rather than shrinking K.
    """
    text = output or ""
    if expansion_count <= 1:
        match = re.search(r"<solution>(.*?)</solution>", text, re.DOTALL)
        body = match.group(1).strip() if match else ""
        return [body] if len(body) >= MIN_SELECTED_SOLUTION_CHARS else []
    solutions = []
    for i in range(1, expansion_count + 1):
        match = re.search(
            rf"<solution_{i}>(.*?)</solution_{i}>", text, re.DOTALL
        )
        body = match.group(1).strip() if match else ""
        if len(body) >= MIN_SELECTED_SOLUTION_CHARS:
            solutions.append(body)
        elif body:
            logger.warning(
                f"[GenericSearch] Selector <solution_{i}> is {len(body)} "
                f"chars (< {MIN_SELECTED_SOLUTION_CHARS}) — rejected as "
                "malformed"
            )
        else:
            logger.warning(f"[GenericSearch] Selector omitted <solution_{i}>")
    if not solutions:
        match = re.search(r"<solution>(.*?)</solution>", text, re.DOTALL)
        body = match.group(1).strip() if match else ""
        if len(body) >= MIN_SELECTED_SOLUTION_CHARS:
            solutions.append(body)
    return solutions

ENSEMBLE_MEMBER_CLIS = frozenset({"claude_code", "codex", "oss_claude_code"})
_ENSEMBLE_MEMBER_KEYS = frozenset({"cli", "model", "effort", "lens"})
# oss_claude_code members additionally carry their endpoint wiring; the
# secret itself stays out of config (auth_token_env is the VAR NAME).
_OSS_MEMBER_KEYS = frozenset({"base_url", "auth_token_env"})


def normalize_ensemble_member(value: Any, role: str) -> Dict[str, str]:
    """Validate one ideation-ensemble member (or selector) config entry."""
    if not isinstance(value, dict):
        raise ValueError(f"{role} must be a mapping, got {type(value).__name__}")
    unknown = sorted(set(value) - _ENSEMBLE_MEMBER_KEYS - _OSS_MEMBER_KEYS)
    if unknown:
        raise ValueError(f"{role} has unknown keys: {', '.join(unknown)}")
    cli = value.get("cli")
    if cli not in ENSEMBLE_MEMBER_CLIS:
        allowed = ", ".join(sorted(ENSEMBLE_MEMBER_CLIS))
        raise ValueError(f"{role}.cli must be one of: {allowed}")
    model = value.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"{role}.model must be a non-empty string")
    oss_keys_present = sorted(_OSS_MEMBER_KEYS & set(value))
    if cli == "oss_claude_code":
        for key in sorted(_OSS_MEMBER_KEYS):
            if not isinstance(value.get(key), str) or not value[key].strip():
                raise ValueError(
                    f"{role}.{key} must be a non-empty string for "
                    "cli=oss_claude_code"
                )
    elif oss_keys_present:
        raise ValueError(
            f"{role} keys {', '.join(oss_keys_present)} are only valid for "
            "cli=oss_claude_code"
        )
    return dict(value)


def normalize_ideation_ensemble(value: Any) -> Optional[List[Dict[str, str]]]:
    """Validate the ideation_ensemble param (None keeps single-session mode)."""
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        raise ValueError(
            "ideation_ensemble must be a non-empty list of member mappings "
            "(omit it entirely for single-session ideation)"
        )
    return [
        normalize_ensemble_member(member, role=f"ideation_ensemble[{i}]")
        for i, member in enumerate(value)
    ]


def normalize_parent_policy(value: Any) -> str:
    """Validate a generic-search parent policy."""
    if not isinstance(value, str) or value not in PARENT_POLICIES:
        allowed = ", ".join(sorted(PARENT_POLICIES))
        raise ValueError(
            f"parent_policy must be one of: {allowed}"
        )
    return value


@dataclass(frozen=True)
class ParentSelection:
    """A branch and node ID selected as one consistent parent."""

    branch_name: str
    node_id: Optional[int]


@register_strategy("generic")
class GenericSearch(SearchStrategy):
    """
    Generic search strategy with Claude Code ideation and implementation.
    
    Each iteration:
    1. Generate a solution using Claude Code + MCP gates (idea, code, research, experiment_history, repo_memory)
    2. Implement and evaluate using Claude Code + MCP gates (code, research, repo_memory)
    3. Generate feedback
    4. Store result and continue
    
    Key features:
    - Claude Code as ideation agent with read-only codebase access
    - Claude Code as implementation agent with full write access
    - MCP gates for external knowledge (wiki_idea_search, wiki_code_search, research_*, experiment_history, repo_memory)
    - RepoMemory access via MCP tools for architecture understanding
    
    Config params:
        - idea_generation_model: Model for solution generation (default: claude-opus-4-5-20251101)
        - implementation_model: Model for implementation (default: claude-opus-4-5-20251101)
        - auth_mode: Claude authentication mode: auto, oauth, api_key, or bedrock
          (default: bedrock, preserving the existing generic strategy behavior)
        - use_bedrock: Deprecated compatibility alias for auth_mode
        - aws_region: AWS region (default: us-east-1)
        - ideation_timeout: Timeout for ideation in seconds (default: 300)
        - implementation_timeout: Timeout for implementation in seconds (default: 600)
        - gate_failure_policy: Missing gate capability behavior: skip, warn, or error
          (default: warn)
        - effort: Optional reasoning effort for both agent sessions
          (low|medium|high|xhigh); None keeps the CLI default
        - ideation_ensemble: Optional list of parallel ideation members,
          each {cli: claude_code|codex|oss_claude_code, model, effort?,
          lens?}; oss_claude_code members additionally require base_url +
          auth_token_env (Anthropic-compatible endpoint + the NAME of the
          env var holding its key). Omit for single-session ideation
          (default)
        - ideation_selector: Required with ideation_ensemble — the
          selector-critic session {cli: claude_code|codex, model, effort?}
        - parent_policy: Parent branch selection: best or baseline (default: best).
          Under `best`, before any validly evaluated node exists, the latest
          committed non-error, non-tampered node is used so in-progress work
          continues in place; `main` only when no committed work exists.
        - ideation_gates: MCP gates for ideation (default: ["research", "experiment_history", "repo_memory", "leeroopedia"])
        - implementation_gates: MCP gates for implementation (default: ["research", "repo_memory", "leeroopedia"])
    """
    
    def __init__(self, config: SearchStrategyConfig, workspace_dir: Optional[str] = None, import_from_checkpoint: bool = False):
        """Initialize generic search strategy."""
        parent_policy = normalize_parent_policy(
            (config.params or {}).get("parent_policy", "best")
        )
        super().__init__(config, workspace_dir, import_from_checkpoint)
        
        # Config params for ideation
        self.idea_generation_model = self.params.get(
            "idea_generation_model", 
            "us.anthropic.claude-opus-4-5-20251101-v1:0"
        )
        if self.params.get("auth_mode") is not None:
            self._claude_auth_settings = {"auth_mode": self.params["auth_mode"]}
        elif "use_bedrock" in self.params:
            # Pass the legacy key through so the adapter can preserve its exact
            # True/False behavior and emit the deprecation warning.
            self._claude_auth_settings = {"use_bedrock": self.params["use_bedrock"]}
        else:
            self._claude_auth_settings = {"auth_mode": "bedrock"}
        self.aws_region = self.params.get("aws_region", "us-east-1")
        self.ideation_timeout = self.params.get("ideation_timeout", 300)
        # Ideation web access: gates the codex member's --search AND the lens
        # planner's WebSearch/WebFetch tools. Default True (Claude members and
        # implementation never had web). Set False for leakage-safe harvest
        # runs on past contests whose reference solutions are published online.
        self.ideation_web_search = self.params.get("web_search", True)
        # Tools to hard-ban on every ideation Claude session when web is off
        # (via --disallowedTools; --allowedTools does not restrict under
        # skip-permissions). Empty when web is on.
        self._web_disallowed_tools = (
            [] if self.ideation_web_search else ["WebSearch", "WebFetch"]
        )
        # Optional reasoning-effort for BOTH agent sessions (ideation and
        # implementation); None keeps the CLI's default.
        self.session_effort = self.params.get("effort")
        # Env vars stripped from every Claude session this strategy spawns
        # (ideation, ensemble members, selector, implementation). Used for
        # credential containment: the orchestrating process may hold a key
        # (e.g. OPENAI_API_KEY for the utility LLM) that agent sessions must
        # not inherit. The codex ideation runner strips its own env.
        self.env_strip = list(self.params.get("env_strip", []))
        # Env defaults for every Claude session (set-if-absent in the child
        # env; ambient wrapper values keep precedence). Carries the Bash-tool
        # clock policy so blocking evaluations are possible (finding 14).
        self.env_defaults = dict(self.params.get("session_env_defaults", {}))
        # Durable-archive recovery root for the registered evaluation (glob
        # of run archive parents, e.g. "tmp/relbench/*/runs"). None disables
        # archive recovery; the live-process wait still applies.
        self.registered_evaluation_archive_glob = self.params.get(
            "registered_evaluation_archive_glob"
        )
        # Optional ensemble ideation: N parallel CLI members + a selector.
        self.ideation_ensemble = normalize_ideation_ensemble(
            self.params.get("ideation_ensemble")
        )
        raw_selector = self.params.get("ideation_selector")
        self.ideation_selector = (
            normalize_ensemble_member(raw_selector, role="ideation_selector")
            if raw_selector is not None
            else None
        )
        if self.ideation_ensemble and self.ideation_selector is None:
            raise ValueError(
                "ideation_ensemble requires an ideation_selector member"
            )
        if self.ideation_selector and self.ideation_selector["cli"] not in (
            "claude_code",
            "codex",
        ):
            raise ValueError(
                "ideation_selector.cli must be claude_code or codex (the "
                "selector reads the worktree to verify candidates)"
            )
        # Optional task-aware lens planning: a web-enabled Claude session
        # designs the member lenses for THIS task at iteration 1, then a
        # keep-or-revise session re-judges the plan against the campaign
        # evidence EVERY later iteration (current plan in
        # .kapso/lens_plan.json, audit trail in lens_plan_history.jsonl).
        # Static member lenses are forbidden while it is enabled.
        self.ideation_lens_planner = normalize_ideation_lens_planner(
            self.params.get("ideation_lens_planner")
        )
        validate_lens_planner_against_ensemble(
            self.ideation_lens_planner, self.ideation_ensemble
        )
        # Design axes of the solution space (anti-freeze contract): the lens
        # replanner must status every axis each iteration and the feedback
        # generator reports the per-axis frontier. Task vocabulary comes
        # from the mode config; the generic default covers common ML axes.
        self.design_axes = normalize_design_axes(
            self.params.get("design_axes")
        )
        # K-way node expansion: the selector emits top-K solutions and each
        # is implemented/evaluated/fed-back on its own branch, in parallel,
        # with a barrier before the next iteration. K=1 is today's path.
        self.node_expansion_value, self.expansion_lane_env = (
            normalize_node_expansion(self.params)
        )
        validate_node_expansion_config(
            self.node_expansion_value,
            self.ideation_ensemble,
            self.ideation_selector,
        )
        # Include experiment_history, repo_memory, and leeroopedia gates by default for ideation
        self.ideation_gates = self.params.get("ideation_gates", ["research", "experiment_history", "repo_memory", "leeroopedia"])
        
        # Config params for implementation
        self.implementation_model = self.params.get(
            "implementation_model",
            "us.anthropic.claude-opus-4-5-20251101-v1:0"
        )
        # Which CLI runs implementation sessions. Both mount the gate MCP
        # servers; codex reports no cost telemetry (ledger undercounts).
        # A crashed implementation session (provider safety classifier, CLI
        # abort) leaves the lane's remaining time unused. When set, a crash
        # retries ONCE on this model — contest 5 (2026-08-06) lost both lanes
        # to codex's "flagged for possible cybersecurity risk" kill on an
        # adversarial-robustness task.
        self.implementation_fallback_model = self.params.get(
            "implementation_fallback_model"
        )
        self.implementation_cli = str(
            self.params.get("implementation_cli", "claude_code")
        )
        if self.implementation_cli not in ("claude_code", "codex"):
            raise ValueError(
                "implementation_cli must be claude_code or codex, got "
                f"{self.implementation_cli!r}"
            )
        self.ideation_candidates_per_member = int(
            self.params.get(
                "ideation_candidates_per_member", ENSEMBLE_CANDIDATES_PER_MEMBER
            )
        )
        if self.ideation_candidates_per_member < 1:
            raise ValueError(
                "ideation_candidates_per_member must be >= 1, got "
                f"{self.ideation_candidates_per_member}"
            )
        self.implementation_timeout = self.params.get("implementation_timeout", 600)
        self.gate_failure_policy = self.params.get("gate_failure_policy", "warn")
        self.implementation_gates = self.params.get("implementation_gates", ["research", "repo_memory", "leeroopedia"])
        self.parent_policy = parent_policy
        
        # Experiment history path (set by orchestrator)
        self.experiment_history_path = self.params.get(
            "experiment_history_path",
            os.path.join(self.workspace_dir, ".kapso", "experiment_history.json")
        )

        # Campaign shared cache: persists across experiments (and campaigns,
        # when params.shared_cache_dir points at a task-level path). Sessions
        # find it via the env var; the registry's artifact offer is rendered
        # into ideation + implementation prompts as OPTIONAL context.
        self.shared_cache_dir, self.shared_artifacts_brief = (
            build_shared_artifacts_brief(
                self.workspace_dir, self.params.get("shared_cache_dir")
            )
        )
        self.env_defaults.setdefault(
            SHARED_CACHE_ENV_VAR, str(self.shared_cache_dir)
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
        print(f"  - idea_generation_model: {self.idea_generation_model}")
        print(f"  - implementation_model: {self.implementation_model}")
        print(f"  - auth: {self._claude_auth_settings}")
        print(f"  - ideation_gates: {self.ideation_gates}")
        print(f"  - implementation_gates: {self.implementation_gates}")
        print(f"  - gate_failure_policy: {self.gate_failure_policy}")
        print(f"  - parent_policy: {self.parent_policy}")
        print(f"  - experiment_history_path: {self.experiment_history_path}")
        print(f"  - feedback_generator: {'configured' if self.feedback_generator else 'not configured'}")
        
        # Initialize workspace with empty main file only for empty workspaces.
        # If the workspace is seeded from an existing repo, we must not overwrite it.
        if workspace_dir is None and not self.workspace.is_seeded:
            self._initialize_workspace()
    
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

    def run(self, context: Any, budget_progress: float = 0.0) -> SearchNode:
        """
        Execute one iteration of generic search.
        
        Node lifecycle:
        1. Generate solution (agent queries experiment history via MCP)
        2. Implement (developer agent handles implementation + evaluation)
        3. Extract results from agent output
        4. Generate feedback
        
        Args:
            context: Either a ContextData object (legacy) or a problem string
            budget_progress: Budget progress percentage (0-100)
        
        Returns:
            SearchNode with solution, evaluation_output, feedback, should_stop
        """
        self.iteration_count += 1
        print(f"\n[GenericSearch] Iteration {self.iteration_count}, budget: {budget_progress:.1f}%")

        # An eval-only VALIDATE grant short-circuits the whole lifecycle:
        # no ideation, no implementation — one full-fidelity measurement of
        # an existing artifact, appended to its node.
        decision = self.fidelity_decision
        if decision is not None and decision.profile == PROFILE_VALIDATE:
            return self._run_validate(decision)
        
        # Extract problem from context (support both string and ContextData)
        if isinstance(context, str):
            problem = context
        else:
            problem = str(getattr(context, "problem", context))
        
        iteration_started_monotonic = time.monotonic()
        iteration_started_at = datetime.now(timezone.utc).isoformat()

        # Select the branch and its node ID once so the recorded lineage, the
        # ideation view, and the implementation base cannot diverge.
        parent = self._select_parent()

        # Step 1: Generate solution(s). With node_expansion_value > 1 the
        # selector emits a ranked top-K; each entry becomes one lane.
        solutions, ideation_sections, ideation_telemetry = self._generate_solution(
            problem,
            parent.branch_name,
        )
        if len(solutions) > 1 and decision is not None and decision is not FULL_PASSTHROUGH:
            raise ValueError(
                "node_expansion_value > 1 is not supported with fidelity "
                "grants active — run with the fidelity block disabled"
            )
        for solution in solutions:
            print(f"[GenericSearch] Generated solution ({len(solution)} chars)")

        return self._expand_round(
            problem=problem,
            solutions=solutions,
            parent=parent,
            decision=decision,
            ideation_sections=ideation_sections,
            ideation_telemetry=ideation_telemetry,
            iteration_started_at=iteration_started_at,
            iteration_started_monotonic=iteration_started_monotonic,
        )

    def _run_expansion_lane(
        self,
        problem: str,
        solution: str,
        node_id: int,
        parent: "ParentSelection",
        decision,
        ideation_sections: List[str],
        ideation_telemetry: Dict[str, float],
        iteration_started_at: str,
        lane_index: int,
    ) -> SearchNode:
        """One lane: node creation -> implementation -> result extraction.

        Feedback/integrity run post-barrier (serialized) so lane threads
        never interleave feedback streams; everything here is lane-local
        except the workspace push, which the repo_lock serializes.
        """
        lane_tag = (
            f"[lane {lane_index}] " if self.node_expansion_value > 1 else ""
        )
        node = SearchNode(
            node_id=node_id,
            parent_node_id=parent.node_id,
            solution=solution,
            workspace_dir=self.workspace_dir,
        )
        node.started_at = iteration_started_at
        node.phase_telemetry["ideation"] = ideation_telemetry
        if decision is not None:
            node.build_fidelity = decision.build_fidelity
            node.eval_fidelity = decision.eval_fidelity
            if decision.profile == "full":
                node.promoted_from = decision.target_node_id

        # Step 2: Implement - developer agent handles everything
        branch_name = f"generic_exp_{node.node_id}"

        print(
            f"{lane_tag}[GenericSearch] Implementing on branch: {branch_name} "
            f"(from {parent.branch_name})"
        )

        agent_output, implementation_telemetry, recovered = self._implement(
            solution=solution,
            problem=problem,
            branch_name=branch_name,
            parent_branch_name=parent.branch_name,
            ideation_repo_memory_sections_consulted=ideation_sections,
            lane_index=lane_index,
        )
        node.phase_telemetry["implementation"] = implementation_telemetry

        # Update node with implementation results
        node.branch_name = branch_name
        node.parent_branch_name = parent.branch_name
        node.agent_output = agent_output
        node.code_diff = self._get_code_diff(branch_name, parent.branch_name)

        # Step 3: Extract results from agent output JSON
        agent_result = self._extract_agent_result(agent_output)

        if agent_result:
            node.code_changes_summary = agent_result.get("code_changes_summary", "")
            node.evaluation_script_path = agent_result.get("evaluation_script_path", "")
            node.technical_difficulties = agent_result.get("technical_difficulties", "")
            node.evaluation_output = agent_result.get("evaluation_output", agent_output)
            # Score from agent result (may be overridden by feedback generator)
            if agent_result.get("score") is not None:
                node.score = float(agent_result.get("score", 0.0))
            print(f"{lane_tag}[GenericSearch] Extracted result from agent JSON")
        else:
            # Fallback: use raw agent output
            node.evaluation_output = agent_output
            print(f"{lane_tag}[GenericSearch] Warning: No JSON result from agent, using raw output")

        # Step 3b: the implementor is the primary author of
        # technical_difficulties; the fallback reconstructs it when the tag
        # is missing. Purely mechanical trigger — never score/outcome-based.
        self._ensure_technical_difficulties(node)

        # Step 3c: inject the manifest recovered from the durable run archive
        # (returned by this lane's own _implement) so the score of record
        # survives a session that died before printing it.
        if recovered and self._manifest_score_of_record(node) is None:
            node.evaluation_output = (
                (node.evaluation_output or "") + "\n" + recovered
            )
        return node

    def _expand_round(
        self,
        problem: str,
        solutions: List[str],
        parent: "ParentSelection",
        decision,
        ideation_sections: List[str],
        ideation_telemetry: Dict[str, float],
        iteration_started_at: str,
        iteration_started_monotonic: float,
    ) -> SearchNode:
        """Implement K solutions on K branches; barrier; feedback in order.

        K=1 preserves today's single-node lifecycle exactly (inline, no
        executor). K>1 allocates node ids/branches up front, fans lanes out
        on threads, then runs integrity+feedback serially in id order and
        appends nodes to history in that order. The returned representative
        is the best-scoring node; its should_stop carries any lane's stop.
        """
        lane_count = len(solutions)
        first_node_id = len(self.node_history)
        lane_args = [
            (solutions[i], first_node_id + i, i) for i in range(lane_count)
        ]

        if lane_count == 1:
            solution, node_id, lane_index = lane_args[0]
            nodes = [
                self._run_expansion_lane(
                    problem, solution, node_id, parent, decision,
                    ideation_sections, ideation_telemetry,
                    iteration_started_at, lane_index,
                )
            ]
        else:
            print(
                f"[GenericSearch] Node expansion: {lane_count} lanes "
                f"(nodes {first_node_id}..{first_node_id + lane_count - 1}) "
                f"from parent {parent.branch_name}"
            )
            with ThreadPoolExecutor(max_workers=lane_count) as executor:
                nodes = list(
                    executor.map(
                        lambda args: self._run_expansion_lane(
                            problem, args[0], args[1], parent, decision,
                            ideation_sections, ideation_telemetry,
                            iteration_started_at, args[2],
                        ),
                        lane_args,
                    )
                )

        # Post-barrier: integrity + feedback serialized in node-id order —
        # deterministic history, no interleaved feedback sessions.
        for node in nodes:
            if self.enforce_evaluation_integrity(node):
                self._generate_feedback(node)
                self._record_evaluation_attempt(node)
            else:
                print(
                    "[GenericSearch] Rejected invalid provided evaluation: "
                    f"{node.evaluation_integrity_error}"
                )
            # Stamp iteration totals: wall-clock for the whole iteration,
            # spend as the sum of attributed phase costs.
            node.duration_seconds = (
                time.monotonic() - iteration_started_monotonic
            )
            node.cost_usd = sum(
                phase.get("cost_usd", 0.0)
                for phase in node.phase_telemetry.values()
            )
            self.node_history.append(node)
            print(
                f"[GenericSearch] ✓ Node {node.node_id} completed: "
                f"score={node.score}, should_stop={node.should_stop}"
            )

        representative = pick_representative(
            nodes, self.problem_handler.maximize_scoring
        )
        if lane_count > 1:
            representative.should_stop = any(n.should_stop for n in nodes)
            print(
                "[GenericSearch] Round winner: node "
                f"{representative.node_id} (score={representative.score}); "
                f"stop={representative.should_stop}"
            )
        return representative

    def _generate_solution(
        self, problem: str, parent_branch: str
    ) -> Tuple[List[str], List[str], Dict[str, float]]:
        """
        Generate solution using Claude Code with MCP gates.
        
        Uses Claude Code as ideation agent with:
        - Read-only access to repo (Read, MCP tools for repo_memory)
        - RepoMemory via CLI
        - Idea/Code/Research/ExperimentHistory gates via MCP
        
        Args:
            problem: Problem description
            parent_branch: Git branch to base ideation on

        Returns:
            Tuple of (solutions, sections_consulted, phase_telemetry) —
            solutions is rank-ordered, length <= node_expansion_value
            (single-session ideation always yields exactly one).
        """
        from kapso.execution.coding_agents.base import CodingAgentConfig
        from kapso.execution.coding_agents.adapters.claude_code_agent import ClaudeCodeCodingAgent
        from kapso.gated_mcp import get_mcp_config
        
        # 1. Load RepoMemory (read-only)
        repo_memory_doc = RepoMemoryManager.load_from_git_branch(
            self.workspace.repo, parent_branch
        ) or {}
        repo_memory_brief = RepoMemoryManager.render_summary_and_toc(
            repo_memory_doc, max_chars=2500
        )
        
        # Materialize the selected ref without changing the root workspace's
        # checkout. Every read-only ideation surface points at this same tree.
        with self.workspace.materialize_ref(parent_branch) as ideation_dir:
            # 2. Configure gates against the selected parent tree. Keep the
            # history path absolute because the MCP process may run elsewhere.
            mcp_servers, mcp_tools = get_mcp_config(
                gates=self.ideation_gates,
                experiment_history_path=os.path.abspath(
                    self.experiment_history_path
                ),
                experiment_embedding_model=(
                    self.llm.resolve_model(None, default_role="embedding")
                    if self.llm is not None
                    else None
                ),
                repo_root=ideation_dir,
                include_base_tools=False,
                gate_failure_policy=self.gate_failure_policy,
            )

            # 3. Build restricted tool set (read-only for ideation). Claude
            # CLIs research with their NATIVE web tools — so WebSearch/
            # WebFetch join the whitelist whenever ideation web access is on
            # (the research_* gate proxies coexist; gates decide availability).
            ideation_allowed_tools = [
                "Read",
                *(["WebSearch", "WebFetch"] if self.ideation_web_search else []),
                *[t for t in mcp_tools if t.startswith("mcp__")],
            ]

            logger.info(
                f"[GenericSearch] Ideation tools: {ideation_allowed_tools}"
            )

            if self.ideation_ensemble:
                return self._generate_solution_ensemble(
                    problem=problem,
                    repo_memory_brief=repo_memory_brief,
                    ideation_dir=ideation_dir,
                    mcp_servers=mcp_servers,
                    ideation_allowed_tools=ideation_allowed_tools,
                )

            # 4. Configure Claude Code for ideation (read-only mode).
            config = CodingAgentConfig(
                agent_type="claude_code",
                model=self.idea_generation_model,
                debug_model=self.idea_generation_model,
                agent_specific={
                    **self._claude_auth_settings,
                    "env_strip": self.env_strip,
                    "env_defaults": self.env_defaults,
                    "aws_region": self.aws_region,
                    "mcp_servers": mcp_servers,
                    "allowed_tools": ideation_allowed_tools,
                    "disallowed_tools": self._web_disallowed_tools,
                    "timeout": self._clamped_timeout(self.ideation_timeout),
                    "streaming": True,
                    "planning_mode": False,
                    "effort": self.session_effort,
                },
            )

            # 5. Build the ideation prompt.
            prompt = self._build_ideation_prompt(
                problem=problem,
                repo_memory_brief=repo_memory_brief,
            )

            # 6. Run Claude Code from the selected parent worktree.
            print("[GenericSearch] Running Claude Code ideation...")
            agent = ClaudeCodeCodingAgent(config)
            agent.initialize(ideation_dir)

            phase_started = time.monotonic()
            try:
                result = agent.generate_code(prompt)
                telemetry = {
                    "cost_usd": agent.get_cumulative_cost(),
                    "duration_seconds": time.monotonic() - phase_started,
                }

                if not result.success:
                    logger.warning(
                        f"[GenericSearch] Ideation failed: {result.error}"
                    )
                    salvaged = self._salvage_ideation_output(result)
                    if salvaged is not None:
                        print(
                            "[GenericSearch] Salvaged partial output "
                            f"({len(salvaged)} chars) from the "
                            "deadline-terminated ideation session"
                        )
                        return (
                            [salvaged],
                            self._extract_sections_consulted(result.output),
                            telemetry,
                        )
                    return [self._fallback_solution(problem)], [], telemetry

                solution = self._extract_solution_from_output(result.output)
                sections_consulted = self._extract_sections_consulted(
                    result.output
                )

                print(
                    "[GenericSearch] Ideation complete, sections consulted: "
                    f"{sections_consulted}"
                )
                return [solution], sections_consulted, telemetry
            finally:
                agent.cleanup()
    
    def _run_lens_planner_session(self, prompt: str, ideation_dir: str):
        """One planner/replanner claude session; returns (result, cost_usd)."""
        return run_lens_planner_session(
            prompt,
            ideation_dir,
            planner=self.ideation_lens_planner,
            claude_auth_settings=self._claude_auth_settings,
            env_strip=self.env_strip,
            env_defaults=self.env_defaults,
            aws_region=self.aws_region,
            web_disallowed_tools=self._web_disallowed_tools,
            ideation_web_search=self.ideation_web_search,
            session_effort=self.session_effort,
            artifacts_dir=self._ideation_artifacts_dir(),
        )

    def _resolve_member_lenses(
        self, problem: str, ideation_dir: str
    ) -> tuple:
        """Task-aware lenses, replanned per iteration; returns (lenses, cost)."""
        if not self.ideation_lens_planner:
            return None, 0.0
        return resolve_member_lenses(
            problem,
            ideation_dir,
            ideation_lens_planner=self.ideation_lens_planner,
            ideation_ensemble=self.ideation_ensemble,
            workspace_dir=self.workspace_dir,
            iteration_count=self.iteration_count,
            shared_artifacts_brief=self.shared_artifacts_brief,
            design_axes=self.design_axes,
            node_history=self.node_history,
            campaign_state_brief=self._campaign_state_brief,
            get_best_experiment=self.get_best_experiment,
            run_planner_session=self._run_lens_planner_session,
        )

    def _generate_solution_ensemble(
        self,
        problem: str,
        repo_memory_brief: str,
        ideation_dir: str,
        mcp_servers: Dict[str, Any],
        ideation_allowed_tools: List[str],
    ) -> Tuple[str, List[str], Dict[str, float]]:
        """Fan out ideation across CLI members, then select one solution.

        Members run in parallel (they are API-bound, never GPU-bound) inside
        the same read-only worktree; a selector-critic session chooses among
        the pooled <solution> candidates. Fail-soft ladder: selector failure
        -> first claude_code candidate -> any candidate -> template fallback.
        """
        phase_started = time.monotonic()

        base_prompt = self._build_ideation_prompt(
            problem=problem, repo_memory_brief=repo_memory_brief
        )
        addendum_template = load_prompt(
            "execution/search_strategies/generic/prompts/ideation_ensemble_addendum.md"
        )

        member_lenses, lens_planner_cost = self._resolve_member_lenses(
            problem, ideation_dir
        )

        # Deadlines are computed AFTER the planner session so its wall time
        # squeezes this iteration's members instead of overflowing the phase.
        clamp = self._clamped_timeout(self.ideation_timeout)
        member_deadline = max(60.0, clamp * ENSEMBLE_MEMBER_TIME_FRACTION)
        selector_deadline = max(
            ENSEMBLE_SELECTOR_MIN_SECONDS, clamp * ENSEMBLE_SELECTOR_TIME_FRACTION
        )

        def run_member(member: Dict[str, str], lens: str) -> Dict[str, Any]:
            prompt = base_prompt + "\n\n" + render_prompt(
                addendum_template,
                {
                    "lens": lens,
                    "candidate_count": str(self.ideation_candidates_per_member),
                },
            )
            label = f"{member['cli']}:{member['model']}"
            print(f"[GenericSearch] Ensemble ideation member starting: {label}")
            # Every member persists its transcript here, not just codex: the
            # claude-driven members used to stream to the console only, so
            # their reasoning survived just in whatever wrapper happened to
            # capture stdout.
            artifacts_dir = self._ideation_artifacts_dir()
            if member["cli"] == "codex":

                def run_codex_once(attempt_deadline: float) -> tuple:
                    return codex_ideation.run_codex_ideation(
                        prompt=prompt,
                        model=member["model"],
                        cwd=ideation_dir,
                        timeout_seconds=attempt_deadline,
                        effort=member.get("effort"),
                        artifacts_dir=artifacts_dir,
                        web_search=self.ideation_web_search,
                    )

                def extract(output: str) -> list:
                    found = re.findall(
                        r"<solution>(.*?)</solution>", output, re.DOTALL
                    )
                    # Echo-drop: anything that appears verbatim in OUR OWN
                    # prompt is the transcript echoing the format example
                    # back (run #8's "blank template" candidate), never a
                    # model contribution.
                    return [
                        c.strip() for c in found
                        if c.strip() and c.strip() not in prompt
                    ]

                output, timed_out, duration, meta = run_codex_once(member_deadline)
                candidates = extract(output)
                if not candidates and not timed_out:
                    # Transient turn failure (run #8 iters 1-2: empty final
                    # message on the first calls after auth shipping). One
                    # retry inside the remaining member window self-heals it.
                    remaining = max(60.0, member_deadline - duration)
                    logger.warning(
                        f"[GenericSearch] member {label} returned no "
                        f"candidates (last_message_empty="
                        f"{meta['last_message_empty']}); retrying once "
                        f"({remaining:.0f}s left). Stream tail: "
                        f"{meta['stream_tail'][-200:]!r}"
                    )
                    output, timed_out, _dur2, meta = run_codex_once(remaining)
                    candidates = extract(output)
                if (
                    not candidates
                    and timed_out
                    and len(output.strip()) >= MIN_IDEATION_SALVAGE_CHARS
                ):
                    candidates = [
                        "# Salvaged from a deadline-terminated ideation session\n"
                        + self._extract_solution_from_output(output.strip())
                    ]
                return {
                    "label": label,
                    "cli": "codex",
                    "candidates": candidates,
                    "sections": [],
                    "cost_usd": 0.0,
                    "duration_seconds": duration,
                    "timed_out": timed_out,
                    "detail": (
                        "last_message_empty" if meta["last_message_empty"] else "ok"
                    ),
                }

            from kapso.execution.coding_agents.base import CodingAgentConfig
            from kapso.execution.coding_agents.adapters.claude_code_agent import ClaudeCodeCodingAgent
            from kapso.execution.coding_agents.adapters.oss_claude_code_agent import OssClaudeCodeCodingAgent

            is_oss = member["cli"] == "oss_claude_code"
            # WebSearch is an Anthropic SERVER-side tool an OSS endpoint
            # cannot serve (Fireworks 400s the request envelope — verified
            # live on kimi-k3-fast, 2026-08-03), so any oss member keeps
            # client-side WebFetch only.
            member_allowed_tools = (
                [t for t in ideation_allowed_tools if t != "WebSearch"]
                if is_oss else ideation_allowed_tools
            )
            agent_specific = {
                "env_strip": self.env_strip,
                "env_defaults": self.env_defaults,
                "aws_region": self.aws_region,
                "mcp_servers": mcp_servers,
                "allowed_tools": member_allowed_tools,
                "disallowed_tools": self._web_disallowed_tools,
                "timeout": member_deadline,
                "streaming": True,
                "planning_mode": False,
                "effort": member.get("effort", self.session_effort),
                "stream_artifact_path": codex_ideation.ideation_stream_path(
                    artifacts_dir, member["cli"], member["model"]
                ),
            }
            if is_oss:
                # Endpoint wiring replaces first-party auth entirely.
                agent_specific["base_url"] = member["base_url"]
                agent_specific["auth_token_env"] = member["auth_token_env"]
            else:
                agent_specific.update(self._claude_auth_settings)
            config = CodingAgentConfig(
                agent_type=member["cli"],
                model=member["model"],
                debug_model=member["model"],
                agent_specific=agent_specific,
            )
            agent_class = OssClaudeCodeCodingAgent if is_oss else ClaudeCodeCodingAgent
            agent = agent_class(config)
            agent.initialize(ideation_dir)
            result = agent.generate_code(prompt)
            cost = agent.get_cumulative_cost()
            agent.cleanup()
            if not result.success:
                logger.warning(
                    f"[GenericSearch] Ensemble member {label} failed: {result.error}"
                )
                salvaged = self._salvage_ideation_output(result)
                candidates = [salvaged] if salvaged is not None else []
            else:
                candidates = [
                    c.strip()
                    for c in re.findall(
                        r"<solution>(.*?)</solution>", result.output, re.DOTALL
                    )
                ] or [self._extract_solution_from_output(result.output)]
            return {
                "label": label,
                "cli": "claude_code",
                "candidates": candidates,
                "sections": self._extract_sections_consulted(result.output),
                "cost_usd": cost,
            }

        members = self.ideation_ensemble
        resolved_lenses = (
            member_lenses
            if member_lenses is not None
            else [m.get("lens", DEFAULT_MEMBER_LENS) for m in members]
        )
        with ThreadPoolExecutor(max_workers=len(members)) as executor:
            member_results = list(
                executor.map(run_member, members, resolved_lenses)
            )

        pool: List[Dict[str, str]] = []
        sections: List[str] = []
        total_cost = lens_planner_cost
        for member_result in member_results:
            total_cost += member_result["cost_usd"]
            for section in member_result["sections"]:
                if section not in sections:
                    sections.append(section)
            kept = 0
            dropped = 0
            for candidate in member_result["candidates"]:
                # Hygiene observed live: skeleton/echo artifacts and
                # duplicated final messages must never reach the selector.
                if is_degenerate_ensemble_candidate(candidate):
                    dropped += 1
                    continue
                if any(candidate == pooled["text"] for pooled in pool):
                    dropped += 1
                    continue
                kept += 1
                pool.append(
                    {"source": member_result["label"], "cli": member_result["cli"], "text": candidate}
                )
            detail = member_result.get("detail", "ok")
            duration = member_result.get("duration_seconds")
            timing = f", {duration:.0f}s" if duration is not None else ""
            print(
                f"[GenericSearch] member {member_result['label']}: "
                f"candidates={kept}/{self.ideation_candidates_per_member} "
                f"(dropped {dropped}){timing}, "
                f"timed_out={member_result.get('timed_out', False)}, {detail}"
            )
            if kept < self.ideation_candidates_per_member:
                logger.warning(
                    f"[GenericSearch] member {member_result['label']} "
                    f"under-delivered: {kept} of "
                    f"{self.ideation_candidates_per_member} candidates"
                )

        telemetry = {
            "cost_usd": total_cost,
            "duration_seconds": time.monotonic() - phase_started,
        }
        print(
            f"[GenericSearch] Ensemble ideation pooled {len(pool)} candidates "
            f"from {len(members)} members"
        )
        if not pool:
            return [self._fallback_solution(problem)], sections, telemetry
        if len(pool) == 1:
            print("[GenericSearch] Single candidate — selector skipped")
            return [pool[0]["text"]], sections, telemetry

        chosen = self._select_from_candidates(
            problem=problem,
            repo_memory_brief=repo_memory_brief,
            pool=pool,
            ideation_dir=ideation_dir,
            selector_deadline=selector_deadline,
        )
        telemetry["cost_usd"] += chosen["cost_usd"]
        telemetry["duration_seconds"] = time.monotonic() - phase_started
        return chosen["solutions"], sections, telemetry

    def _campaign_state_brief(self) -> str:
        """Factual campaign trajectory for the selector's return judgment.

        The selector prompt values candidates by expected return against the
        GOAL's bar; this supplies the other half of that arithmetic — where
        the campaign currently stands and whether progress has stalled.
        """
        scored = [
            n
            for n in self.node_history
            if not n.had_error and n.evaluation_valid and n.score is not None
        ]
        if not scored:
            return (
                "No scored experiments yet — the pool below is the campaign's "
                "first swing; judge return against the GOAL's published bar."
            )
        maximize = self.problem_handler.maximize_scoring
        champion = max(n.score for n in scored) if maximize else min(
            n.score for n in scored
        )
        recent = [float(f"{n.score:.6g}") for n in scored[-5:]]
        stagnation = 0
        running_best: Optional[float] = None
        for node in scored:
            improved = running_best is None or (
                node.score > running_best if maximize else node.score < running_best
            )
            if improved:
                running_best = node.score
                stagnation = 0
            else:
                stagnation += 1
        return (
            f"Scored experiments: {len(scored)}; champion score: {champion:.6g}; "
            f"last {len(recent)} scores: {recent}; consecutive experiments "
            f"without strict improvement: {stagnation}. The GOAL above states "
            "the published bar — judge every candidate's return against the "
            "remaining gap to THAT bar, not against the champion."
        )

    def _select_from_candidates(
        self,
        problem: str,
        repo_memory_brief: str,
        pool: List[Dict[str, str]],
        ideation_dir: str,
        selector_deadline: float,
    ) -> Dict[str, Any]:
        """Run the selector-critic session over the pooled candidates.

        Returns {"solutions": List[str] (rank order, len<=node_expansion_value),
        "cost_usd": float}. With expansion 1 this is today's single pick.
        """
        from kapso.execution.coding_agents.base import CodingAgentConfig
        from kapso.execution.coding_agents.adapters.claude_code_agent import ClaudeCodeCodingAgent

        expansion = self.node_expansion_value
        candidates_block = "\n\n".join(
            f"### Candidate {i} (from {c['source']})\n{c['text']}"
            for i, c in enumerate(pool, 1)
        )
        prompt = render_prompt(
            load_prompt(
                "execution/search_strategies/generic/prompts/ideation_selector.md"
            ),
            {
                "problem": problem,
                "repo_memory_brief": repo_memory_brief
                or "(No repo memory available)",
                "campaign_state": self._campaign_state_brief(),
                "candidates": candidates_block,
            },
        )
        if expansion > 1:
            # Appended override keeps the K=1 selector prompt byte-identical.
            prompt += "\n\n" + render_prompt(
                load_prompt(
                    "execution/search_strategies/generic/prompts/"
                    "ideation_selector_expansion_addendum.md"
                ),
                {"expansion_count": str(expansion)},
            )
        selector = self.ideation_selector

        def run_selector_session(session_prompt: str):
            """One selector invocation -> (CodingResult, cost_usd).

            A closure so the malformed-emission retry re-runs the identical
            session with an added corrective instruction.
            """
            if selector["cli"] == "codex":
                from kapso.execution.search_strategies.generic.codex_ideation import (
                    run_codex_ideation,
                )
                from kapso.execution.coding_agents.base import CodingResult

                # web on: selection verifies candidate claims — a cited repo,
                # pretrained model, or dataset must exist and plausibly do what
                # the candidate says before it can win selection.
                # Artifacts go to the WORKSPACE, not ideation_dir: the latter is
                # a materialized ref released after the phase, which silently
                # discarded every selector transcript.
                output, timed_out, _duration, _meta = run_codex_ideation(
                    prompt=session_prompt,
                    model=selector["model"],
                    cwd=ideation_dir,
                    timeout_seconds=selector_deadline,
                    effort=selector.get("effort"),
                    artifacts_dir=self._ideation_artifacts_dir(),
                    web_search=True,
                )
                return CodingResult(
                    success=not timed_out and bool(output.strip()),
                    output=output,
                    error="selector session timed out" if timed_out else None,
                ), 0.0
            config = CodingAgentConfig(
                agent_type="claude_code",
                model=selector["model"],
                debug_model=selector["model"],
                agent_specific={
                    **self._claude_auth_settings,
                    "env_strip": self.env_strip,
                    "env_defaults": self.env_defaults,
                    "aws_region": self.aws_region,
                    "allowed_tools": ["Read", "WebSearch", "WebFetch"],
                    "timeout": selector_deadline,
                    "streaming": True,
                    "planning_mode": False,
                    "effort": selector.get("effort", self.session_effort),
                    "stream_artifact_path": codex_ideation.ideation_stream_path(
                        self._ideation_artifacts_dir(), "selector",
                        selector["model"],
                    ),
                },
            )
            agent = ClaudeCodeCodingAgent(config)
            agent.initialize(ideation_dir)
            session_result = agent.generate_code(session_prompt)
            session_cost = agent.get_cumulative_cost()
            agent.cleanup()
            return session_result, session_cost

        result, cost = run_selector_session(prompt)

        reasoning = re.search(
            r"<selection_reasoning>(.*?)</selection_reasoning>",
            result.output or "",
            re.DOTALL,
        )
        if reasoning:
            print(
                "[GenericSearch] Selector reasoning:\n"
                + reasoning.group(1).strip()
            )
        solutions = (
            parse_selected_solutions(result.output, expansion)
            if result.success
            else []
        )
        if len(solutions) < expansion:
            # One retry before topping up: a malformed emission is usually a
            # one-off, and a re-ranked full set beats raw pool order.
            logger.warning(
                f"[GenericSearch] Selector returned {len(solutions)}/"
                f"{expansion} usable solutions — retrying once"
            )
            retry_result, retry_cost = run_selector_session(
                prompt
                + f"\n\nYour previous response did not emit {expansion} "
                f"complete <solution_N> blocks. Emit ALL {expansion}, each a "
                "full implementation spec of its own — never a placeholder, "
                "a fragment, or a few words."
            )
            cost += retry_cost
            retry_solutions = (
                parse_selected_solutions(retry_result.output, expansion)
                if retry_result.success
                else []
            )
            if len(retry_solutions) > len(solutions):
                solutions = retry_solutions
        if len(solutions) < expansion:
            # Never shrink the round: the pooled candidates are full specs, so
            # top up rank order (claude first) with texts not already selected.
            logger.warning(
                f"[GenericSearch] Selector still short ({len(solutions)}/"
                f"{expansion}) — topping up from the pooled candidates"
            )
            ordered = [c for c in pool if c["cli"] == "claude_code"] + [
                c for c in pool if c["cli"] != "claude_code"
            ]
            for candidate in ordered:
                if len(solutions) >= expansion:
                    break
                if candidate["text"] not in solutions:
                    solutions.append(candidate["text"])
        return {"solutions": solutions, "cost_usd": cost}

    def _build_ideation_prompt(
        self,
        problem: str,
        repo_memory_brief: str,
    ) -> str:
        """Build the ideation prompt for Claude Code."""
        # Load and render the prompt template
        template = load_prompt("execution/search_strategies/generic/prompts/ideation_claude_code.md")
        return render_prompt(
            template,
            {
                "problem": problem or "(No problem description provided)",
                "repo_memory_brief": repo_memory_brief or "(No repo memory available)",
                "budget_status": self._render_budget_status(),
                "shared_artifacts_brief": self.shared_artifacts_brief,
            },
        )
    
    def _extract_solution_from_output(self, output: str) -> str:
        """Extract solution from Claude Code output."""
        # Look for <solution>...</solution> tags
        match = re.search(r'<solution>(.*?)</solution>', output, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Fallback: look for markdown headers that indicate a solution
        # Try to find "# Core Idea" section
        core_idea_match = re.search(r'#\s*Core Idea.*', output, re.DOTALL)
        if core_idea_match:
            return core_idea_match.group(0).strip()
        
        # Last resort: return entire output (may contain useful info)
        logger.warning("[GenericSearch] Could not extract solution tags, using full output")
        return output
    
    def _extract_sections_consulted(self, output: str) -> List[str]:
        """Extract RepoMemory sections consulted from Claude Code output."""
        # Look for repo_memory cli get-section calls
        sections = re.findall(r'repo_memory\.cli\s+get-section\s+(\S+)', output)
        # Also look for direct section references in tool calls
        sections.extend(re.findall(r'get-section\s+["\']?(\S+?)["\']?\s', output))
        # Deduplicate while preserving order
        seen = set()
        result = []
        for s in sections:
            # Clean up section ID (remove quotes, trailing punctuation)
            s = s.strip('"\'.,;:')
            if s and s not in seen:
                seen.add(s)
                result.append(s)
        return result
    
    def _salvage_ideation_output(self, result) -> Optional[str]:
        """Recover a deadline-terminated ideation's partial output.

        Only deadline kills are salvageable: the session was mid-work and
        its streamed text is the research and draft plan produced so far —
        discarding it forces the next phase to redo that work (a live run
        lost 30 minutes of research exactly this way). Non-deadline
        failures keep the fallback path: their output is error noise, not
        a plan.
        """
        if not result.metadata.get("deadline_exceeded"):
            return None
        partial = (result.output or "").strip()
        if len(partial) < MIN_IDEATION_SALVAGE_CHARS:
            return None
        return (
            "# Salvaged from a deadline-terminated ideation session\n"
            "The ideation agent hit its deadline before emitting a final "
            "solution. The notes below are its partial output: treat them "
            "as research findings plus a draft plan, and turn them into an "
            "implementation directly instead of re-deriving them.\n\n"
            f"{self._extract_solution_from_output(partial)}"
        )

    def _fallback_solution(self, problem: str) -> str:
        """Generate a fallback solution when Claude Code ideation fails."""
        return f"""# Core Idea
Implement a baseline solution for the given problem.

# Solution Steps
1. Analyze the problem requirements
2. Implement a straightforward solution
3. Add basic error handling
4. Create evaluation metrics

# Hyperparameters
- Use default values from the problem description

# Rationale
Fallback solution due to ideation failure. Focus on correctness over optimization.

Problem: {problem}"""

    def _implement(
        self,
        solution: str,
        problem: str,
        branch_name: str,
        parent_branch_name: str = "main",
        ideation_repo_memory_sections_consulted: Optional[List[str]] = None,
        lane_index: int = 0,
    ) -> Tuple[str, Dict[str, float], Optional[str]]:
        """Implementation session with MCP gates; returns (agent output,
        phase telemetry, recovered manifest line)."""
        lane_env = lane_env_overlay(self.expansion_lane_env, lane_index)
        return run_implementation(
            solution=solution,
            problem=problem,
            branch_name=branch_name,
            parent_branch_name=parent_branch_name,
            ideation_repo_memory_sections_consulted=(
                ideation_repo_memory_sections_consulted
            ),
            lane_index=lane_index,
            workspace=self.workspace,
            llm=self.llm,
            registered_evaluation_manifest=self.registered_evaluation_manifest,
            sync_registered_evaluation=self._sync_registered_evaluation,
            implementation_gates=self.implementation_gates,
            gate_failure_policy=self.gate_failure_policy,
            implementation_cli=self.implementation_cli,
            implementation_model=self.implementation_model,
            implementation_fallback_model=self.implementation_fallback_model,
            claude_auth_settings=self._claude_auth_settings,
            env_strip=self.env_strip,
            env_defaults=self.env_defaults,
            aws_region=self.aws_region,
            lane_env=lane_env,
            session_effort=self.session_effort,
            clamped_timeout=self._clamped_timeout,
            implementation_timeout=self.implementation_timeout,
            session_stream_path=self._session_stream_path,
            build_prompt=self._build_implementation_prompt,
            previous_errors_text="\n".join(
                str(e)
                for e in self.previous_errors[-self.recent_error_count:]
            ),
            lane_brief=render_lane_brief(
                lane_index, self.node_expansion_value, lane_env
            ),
            note_session_started=(
                lambda: setattr(
                    self, "_last_session_started_ts", time.time()
                )
            ),
            note_session_end_facts=(
                lambda facts: setattr(
                    self, "_pending_session_end_facts", facts
                )
            ),
            await_registered_evaluation=self._await_registered_evaluation,
        )

    def _build_implementation_prompt(
        self,
        solution: str,
        problem: str,
        branch_name: str,
        repo_memory_brief: str,
        repo_memory_detail_access_instructions: str,
        previous_errors: str,
        lane_brief: str = "",
    ) -> str:
        """Build the implementation prompt for Claude Code."""
        return build_implementation_prompt(
            solution=solution,
            problem=problem,
            branch_name=branch_name,
            repo_memory_brief=repo_memory_brief,
            repo_memory_detail_access_instructions=(
                repo_memory_detail_access_instructions
            ),
            previous_errors=previous_errors,
            budget_status=self._render_budget_status(),
            evaluation_instructions=self._evaluation_instructions(),
            shared_artifacts_brief=self.shared_artifacts_brief,
            lane_brief=lane_brief,
        )

    def _manifest_of_record(self, node: SearchNode) -> Optional[Dict[str, Any]]:
        """The granted-class manifest from the session's last manifest line."""
        if not self.registered_evaluation_command:
            return None
        return manifest_of_record(
            node,
            registered_evaluation_command=self.registered_evaluation_command,
            fidelity_decision=self.fidelity_decision,
            registered_subsample_seed=self.registered_subsample_seed,
        )

    def _manifest_score_of_record(self, node: SearchNode) -> Optional[float]:
        """The granted-class score from the session's last manifest line."""
        if not self.registered_evaluation_command:
            return None
        return manifest_score_of_record(
            node,
            registered_evaluation_command=self.registered_evaluation_command,
            fidelity_decision=self.fidelity_decision,
            registered_subsample_seed=self.registered_subsample_seed,
        )

    def _record_evaluation_attempt(self, node: SearchNode) -> None:
        """Append the node's measurement under the registered evaluator."""
        if (
            not self.registered_evaluator_id
            or node.score is None
            or node.had_error
            or not node.evaluation_valid
        ):
            return
        record_evaluation_attempt(
            node,
            registered_evaluator_id=self.registered_evaluator_id,
            fidelity_decision=self.fidelity_decision,
            registered_subsample_seed=self.registered_subsample_seed,
            workspace=self.workspace,
        )

    def _execute_registered_evaluation(
        self,
        target: SearchNode,
        *,
        fidelity: str,
        fraction: float,
        deadline_seconds: Optional[float],
    ) -> Optional[float]:
        """Frame-run the registered evaluation on an existing artifact."""
        return execute_registered_evaluation(
            target,
            fidelity=fidelity,
            fraction=fraction,
            deadline_seconds=deadline_seconds,
            registered_evaluator_id=self.registered_evaluator_id,
            registered_subsample_seed=self.registered_subsample_seed,
            registered_data_manifest=self.registered_data_manifest,
            workspace=self.workspace,
            workspace_dir=self.workspace_dir,
            record_eval_duration=self.record_eval_duration,
        )

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

    def refresh_score_projections(
        self, comparability: ComparabilityClass
    ) -> None:
        """Re-project every node's score under one canonical ruler.

        The selectors stay dumb: after an evaluator transition, nodes never
        measured under the new ruler project None — and None never wins.
        """
        for node in self.node_history:
            node.score = project_score(node, comparability)

    def _sync_registered_evaluation(self, session_folder: str) -> None:
        """Overwrite the session's evaluation tree with the registered one."""
        sync_registered_evaluation(session_folder, self.workspace_dir)

    def _evaluation_instructions(self) -> str:
        """Registered-evaluation contract when a maintainer owns evaluation;
        the historical build-your-own instructions otherwise."""
        return evaluation_instructions(self.registered_evaluation_command)

    def _ensure_technical_difficulties(self, node) -> None:
        """Run the fallback reconstruction when the implementor's
        technical_difficulties tag is missing (purely mechanical trigger —
        never score/outcome-based)."""
        if (node.technical_difficulties or "").strip():
            return
        ensure_technical_difficulties(
            node,
            implementation_model=self.implementation_model,
            claude_auth_settings=self._claude_auth_settings,
            aws_region=self.aws_region,
            env_strip=self.env_strip,
            session_effort=self.session_effort,
            clamped_timeout=self._clamped_timeout,
            ideation_timeout=self.ideation_timeout,
            workspace_dir=self.workspace_dir,
            session_stream_path=self._session_stream_path,
        )

    def _await_registered_evaluation(self, output_text: str):
        """Teardown guard for the registered evaluation: wait for a live
        grader, then recover the manifest line from the durable run archive
        (must run BEFORE finalize_session's rmtree)."""
        return await_registered_evaluation(
            output_text,
            registered_evaluation_command=self.registered_evaluation_command,
            registered_evaluation_archive_glob=(
                self.registered_evaluation_archive_glob
            ),
            clamped_timeout=self._clamped_timeout,
            implementation_timeout=self.implementation_timeout,
            session_started_ts=getattr(
                self, "_last_session_started_ts", 0.0
            ),
        )

    def _ideation_artifacts_dir(self) -> str:
        """Where this iteration's ideation transcripts live (lens planner,
        every ensemble member, selector). Under the workspace, so they survive
        the materialized ref the phase runs in."""
        return os.path.join(
            self.workspace_dir, ".kapso", "ideation",
            f"iter{self.iteration_count}",
        )

    def _session_stream_path(self, branch_name: str) -> str:
        """Per-session stream artifact location (survives session kills)."""
        stream_dir = os.path.join(
            self.workspace_dir, ".kapso", "sessions", branch_name
        )
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
        if (
            snapshot.time_budget_seconds is None
            and snapshot.cost_budget_usd is None
        ):
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

    def _select_parent(self) -> ParentSelection:
        """Select one consistent parent according to the configured policy."""
        if self.parent_policy == "baseline":
            return ParentSelection(branch_name="main", node_id=None)

        best = self.get_best_experiment()
        if best is not None:
            return ParentSelection(
                branch_name=best.branch_name,
                node_id=best.node_id,
            )

        # No validly evaluated node exists yet. Committed-but-unevaluated
        # work (a deadline-killed implementation, an evaluation that never
        # ran) is still real progress; branching from `main` strands it on
        # its branch and the next iteration redoes it. Integrity-flagged
        # candidates stay excluded — never build on an evaluator tamperer.
        committed = [
            node
            for node in self.node_history
            if not node.had_error
            and not node.evaluation_integrity_error
            and node.code_diff.strip()
            and node.branch_name
        ]
        if committed:
            latest = max(committed, key=lambda node: node.node_id)
            return ParentSelection(
                branch_name=latest.branch_name,
                node_id=latest.node_id,
            )
        return ParentSelection(branch_name="main", node_id=None)

    def get_experiment_history(self, best_last: bool = False) -> List[SearchNode]:
        """Return all nodes, optionally sorted by score (unscored sort worst)."""
        if best_last:
            return sorted(
                self.node_history,
                key=lambda node: (
                    not node.had_error and node.evaluation_valid and node.score is not None,
                    0.0 if node.score is None
                    else (node.score if self.problem_handler.maximize_scoring else -node.score),
                )
            )
        return self.node_history

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
            key=lambda x: x.score if self.problem_handler.maximize_scoring else -x.score
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
        """Judge one node with the FeedbackGenerator (updates the node
        in-place with feedback, score, and should_stop)."""
        return generate_feedback(
            node,
            feedback_generator=self.feedback_generator,
            goal=self.goal,
            design_axes=design_axes_brief(self.design_axes),
            session_end_facts=getattr(
                self, "_pending_session_end_facts", ""
            ),
            clamped_timeout=self._clamped_timeout,
            manifest_of_record=self._manifest_of_record,
            finalize_run_selection=(
                lambda manifest, valid:
                self.problem_handler.finalize_run_selection(manifest, valid)
            ),
        )

    def _extract_agent_result(self, agent_output: str) -> dict:
        """Extract the structured XML-tag result from agent output
        (empty dict when nothing parses)."""
        return extract_agent_result(agent_output)

    # =========================================================================
    # Checkpoint Methods
    # =========================================================================

    def dump_state(self) -> Dict[str, Any]:
        """Return JSON-compatible generic-search state."""
        return {
            "node_history": [node.to_dict() for node in self.node_history],
            "iteration_count": self.iteration_count,
            "previous_errors": list(self.previous_errors),
            "parent_policy": getattr(self, "parent_policy", "best"),
            "evaluation_integrity": (
                self.dump_evaluation_integrity_state()
            ),
            "scores_evaluator_id": self.scores_evaluator_id,
            "evaluator_transition": self.evaluator_transition,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Restore generic-search state from a versioned run checkpoint."""
        if not isinstance(state, dict):
            raise ValueError("GenericSearch checkpoint state must be an object")
        raw_history = state.get("node_history")
        if not isinstance(raw_history, list):
            raise ValueError(
                "GenericSearch checkpoint node_history must be a list"
            )
        self.node_history = [
            SearchNode.from_dict(node_data) for node_data in raw_history
        ]
        node_ids = [node.node_id for node in self.node_history]
        if node_ids != list(range(len(self.node_history))):
            raise ValueError(
                "GenericSearch checkpoint node IDs must be unique, ordered, "
                "and contiguous from zero"
            )

        iteration_count = state.get(
            "iteration_count", len(self.node_history)
        )
        if (
            isinstance(iteration_count, bool)
            or not isinstance(iteration_count, int)
            or iteration_count < 0
        ):
            raise ValueError(
                "GenericSearch checkpoint iteration_count must be non-negative"
            )
        # No cross-check against node_history: one iteration legitimately
        # spawns several lane nodes, so nodes routinely outnumber
        # iterations. The old "every node consumed an iteration" invariant
        # made every multi-lane checkpoint unrestorable (first live hit:
        # rel-event/user-ignore resume, 2026-08-09 — 4 lane nodes,
        # iteration_count 1, RunCheckpointCorruptError).
        self.iteration_count = iteration_count

        previous_errors = state.get("previous_errors", [])
        if not isinstance(previous_errors, list) or not all(
            isinstance(error, str) for error in previous_errors
        ):
            raise ValueError(
                "GenericSearch checkpoint previous_errors must be strings"
            )
        self.previous_errors = list(previous_errors)

        saved_parent_policy = normalize_parent_policy(
            state.get("parent_policy", "best")
        )
        configured_parent_policy = getattr(
            self,
            "parent_policy",
            saved_parent_policy,
        )
        if saved_parent_policy != configured_parent_policy:
            raise ValueError(
                "GenericSearch checkpoint parent_policy does not match "
                "the configured policy"
            )
        self.parent_policy = saved_parent_policy

        nodes_by_id = {node.node_id: node for node in self.node_history}
        for node in self.node_history:
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
                    "GenericSearch checkpoint parent node and branch do not "
                    "match"
                )
        self.load_evaluation_integrity_state(
            state.get("evaluation_integrity")
        )

        scores_evaluator_id = state.get("scores_evaluator_id", "")
        if not isinstance(scores_evaluator_id, str):
            raise ValueError(
                "GenericSearch checkpoint scores_evaluator_id must be a "
                "string"
            )
        self.scores_evaluator_id = scores_evaluator_id
        transition = state.get("evaluator_transition")
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
            raise ValueError(
                "GenericSearch checkpoint evaluator_transition is invalid"
            )
        self.evaluator_transition = transition
