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

import math
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pathlib import Path
from kapso.core.config import load_config
from kapso.gated_mcp.presets import HOSTED_GATE, knowledge_tools_block, user_setup_gaps
from kapso.execution.search_strategies.base import (
    SearchStrategy,
    SearchStrategyConfig,
    SearchNode,
)
from kapso.execution.search_strategies.factory import register_strategy
from kapso.execution.evaluation_archive import label_run_of_record
from kapso.execution.cost_estimator import (
    DEFAULT_COST_CONFIG,
    normalize_cost_config,
    selection_score,
    validate_cost_value,
)
from kapso.execution.fidelity import (
    FULL_PASSTHROUGH,
    PROFILE_VALIDATE,
    ComparabilityClass,
    FidelityDecision,
    project_score,
    select_committed_candidate,
)
from concurrent.futures import ThreadPoolExecutor
from kapso.execution.search_strategies.generic.expansion_lanes import (
    lane_env_overlay,
    normalize_node_expansion,
    pick_representative,
    render_lane_brief,
    validate_node_expansion_config,
)
from kapso.execution.search_strategies.generic.feedback_generator import FeedbackGenerator
from kapso.execution.search_strategies.generic.feedback_flow import (
    extract_agent_result,
    generate_feedback,
    measured_cost_from_output,
)
from kapso.execution.search_strategies.generic.ideation import (
    build_ideation_prompt,
    campaign_state_brief,
    generate_solution,
    generate_solution_ensemble,
    normalize_ensemble_time_split,
    salvage_ideation_output,
    select_from_candidates,
)
from kapso.execution.search_strategies.generic.implementation import (
    Continuation,
    build_implementation_prompt,
    ensure_technical_difficulties,
    render_follow_up,
    render_inbox_ideation,
    run_implementation,
)
from kapso.execution.inbox import (
    InboxOpenError,
    Request,
    all_answered,
    inbox_path,
    load_requests,
    read_launch_record,
    record_continued,
    requests_for_ids,
)
from kapso.core.prompt_loader import load_prompt, optional_block, render_prompt
from kapso.execution.coding_agents.factory import CodingAgentFactory
from kapso.harness.staging import TASK_STATEMENT_FILE, StagedHarness, host_for, stage_harness
from kapso.harness.verdict import VERDICT_FILE, is_deliverable, is_void, read_verdict, unverified, verification_dir
from kapso.execution.search_strategies.generic.lens_planning import (
    axes_brief,
    current_plan_axes,
    resolve_member_lenses,
    run_lens_planner_session,
    validate_lens_planner,
)
from kapso.core.roles import (
    SELECTOR_CLIS,
    check_params,
    resolve_members,
    resolve_optional,
    resolve_role,
)
from kapso.execution.search_strategies.generic.registered_evaluation import (
    await_registered_evaluation,
    deferred_evaluation_instructions,
    evaluation_instructions,
    execute_candidate_evaluation,
    execute_registered_evaluation,
    manifest_of_record,
    record_evaluation_attempt,
    sync_registered_evaluation,
)
from kapso.execution.search_strategies.generic.shared_cache import (
    SHARED_CACHE_ENV_VAR,
    build_shared_artifacts_brief,
)


PARENT_POLICIES = frozenset({"best", "baseline", "weighted"})
DEFAULT_PARENT_SAMPLING = load_config()["search_strategy"]["params"][
    "parent_sampling"
]


def normalize_parent_policy(value: Any) -> str:
    """Validate a generic-search parent policy."""
    if not isinstance(value, str) or value not in PARENT_POLICIES:
        allowed = ", ".join(sorted(PARENT_POLICIES))
        raise ValueError(
            f"parent_policy must be one of: {allowed}"
        )
    return value


def normalize_parent_sampling(config: Any) -> Dict[str, Any]:
    """Resolve the weighted-policy settings over the config defaults."""
    if config is not None and not isinstance(config, dict):
        raise ValueError("parent_sampling must be a mapping")
    values = dict(DEFAULT_PARENT_SAMPLING)
    if config is not None:
        unknown = set(config) - values.keys()
        if unknown:
            raise ValueError(f"Unknown parent_sampling keys: {sorted(unknown)}")
        values.update(config)
    for name in ("temperature", "offspring_penalty"):
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"parent_sampling.{name} must be a number")
        if not math.isfinite(value):
            raise ValueError(f"parent_sampling.{name} must be finite")
    if values["temperature"] <= 0:
        raise ValueError("parent_sampling.temperature must be positive")
    if values["offspring_penalty"] < 0:
        raise ValueError("parent_sampling.offspring_penalty must be non-negative")
    if isinstance(values["seed"], bool) or not isinstance(values["seed"], int):
        raise ValueError("parent_sampling.seed must be an integer")
    return values


def parent_weights(
    children: Sequence[int],
    temperature: float,
    offspring_penalty: float,
) -> List[float]:
    """Sampling weights for candidates listed best first, with each one's
    child count. Rank, not raw score, keeps the weight scale-free."""
    return [
        math.exp(-rank / temperature) / (1 + offspring_penalty * count)
        for rank, count in enumerate(children)
    ]


def rank_candidates(
    nodes: Sequence[SearchNode], selection_config: Dict[str, Any]
) -> List[SearchNode]:
    """Scored candidates best first, as the campaign ranks them. A node
    whose evaluation never completed (score=None) never ranks; on
    minimize metrics it would otherwise key as 0 and out-rank every real
    score. A node the judge never read ranks nowhere either."""
    scored = [
        node
        for node in nodes
        if not node.had_error and node.evaluation_valid and node.score is not None
        and not node.unjudged
    ]
    return sorted(
        scored,
        key=lambda node: selection_score(node, selection_config),
        reverse=True,
    )


def select_deliverable(
    nodes: Sequence[SearchNode],
    *,
    evaluator_id: str,
    selection_config: Dict[str, Any],
) -> Optional[SearchNode]:
    """The committed-slot winner: evidence tiers, never raw scores.

    Parent selection explores on projected scores (the four-bests split);
    the deliverable follows the tier walk under the registered evaluator,
    so an unvalidated fast leader cannot displace a full-tier candidate at
    delivery. Without registered evidence the score leader stands.
    """
    if evaluator_id and not selection_config["cost"]["enabled"]:
        committed = select_committed_candidate(
            nodes,
            evaluator_id=evaluator_id,
            maximize=selection_config["maximize_scoring"],
        )
        if committed is not None:
            return committed
    ranked = rank_candidates(nodes, selection_config)
    return ranked[0] if ranked else None


@dataclass(frozen=True)
class ParentSelection:
    """A branch and node ID selected as one consistent parent."""

    branch_name: str
    node_id: Optional[int]


def resolve_inbox_settings(
    inbox: Optional[Dict[str, Any]],
    node_expansion_value: int,
    implementation_gates: List[str],
    workspace_dir: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """The campaign's effective inbox settings. On with one implementation
    lane, the request_from_user gate is mounted for implementation
    sessions and the inbox file lives at <workspace>/.kapso/inbox.jsonl
    unless the settings name a path; with parallel lanes the inbox is off
    for this campaign (one printed line — lanes come later)."""
    if not inbox or not inbox.get("enabled"):
        return inbox
    if node_expansion_value > 1:
        print(
            "[GenericSearch] Inbox off for this campaign: node expansion above "
            "1 is not supported by the inbox yet"
        )
        return {**inbox, "enabled": False}
    if "inbox" not in implementation_gates:
        implementation_gates.append("inbox")
    resolved = dict(inbox)
    if not resolved.get("path"):
        if not workspace_dir:
            raise ValueError("an enabled inbox needs a path or a workspace_dir")
        resolved["path"] = str(inbox_path(workspace_dir))
    # Where a value goes: the .env the launch loaded (the launch record;
    # empty for a campaign started without one), named in the prompt so a
    # fix points at a file that survives the session (live L1 run 2).
    if "dotenv_path" not in resolved:
        record = read_launch_record(workspace_dir) if workspace_dir else None
        resolved["dotenv_path"] = record.get("dotenv_path", "") if record else ""
    return resolved


# The harness's verification session (kapso.harness.verdict): its prompt,
# and how many times a session that wrote no verdict runs before the
# attempt is left unverified.
VERIFICATION_PROMPT = "execution/search_strategies/generic/prompts/verification.md"


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
    
    Config (`search_strategy.params`, every key in the packaged config,
    refused by name otherwise — kapso.core.roles):
        - host: the CLI every session runs on unless its role names one;
          the config's `hosts` table gives each CLI its model and its own
          settings (auth_mode, effort, vertex_project, …)
        - ideation, implementation, judge: the roles, each {cli, model,
          effort, timeout} over the host's values; ideation also carries
          `gates`, `candidates_per_member`, `ensemble` (parallel members
          [{cli, model, effort, lens}, …]), `selector`, `lens_planner`
          ({cli, model, effort, timeout}) and `ensemble_time_split`;
          implementation also `gates` and `fallback_model` (a second model
          tried once when the session crashes). A timeout is the session's
          deadline in seconds; None means the budget's clamp alone.
        - gate_failure_policy: a gate the machine cannot serve: skip, warn or error
        - parent_policy: best (the champion) or baseline
        - node_expansion_value, expansion_lane_env: parallel implementation
          lanes per iteration and their environments
        - env_strip, session_env_defaults: every session's environment
        - shared_cache_dir, registered_evaluation_archive_glob,
          repo_memory_failure_policy, repo_memory_max_retries, cost
    The engine threads the rest at launch (bank_serving, experience_intro_path,
    kg_index_path, hosted, inbox, experiment_history_path).
    """

    def __init__(self, config: SearchStrategyConfig, workspace_dir: Optional[str] = None, import_from_checkpoint: bool = False):
        """Initialize generic search strategy."""
        params = config.params or {}
        check_params(params, engine=True)
        parent_policy = normalize_parent_policy(params["parent_policy"])
        self.cost_config = normalize_cost_config(params.get("cost"))
        self.parent_sampling = normalize_parent_sampling(params["parent_sampling"])
        self.before_candidate_evaluation = None
        # How each node's build session ended, keyed by node id: the lanes
        # of one round run at once and each judge reads only its own node's.
        self._session_end_facts: Dict[int, str] = {}
        super().__init__(config, workspace_dir, import_from_checkpoint)

        # The sessions by role over the hosts table (roles.py): ideation,
        # implementation (the verification, the harness setup and the
        # difficulties reconstruction follow it) and the judge, which the
        # orchestrator builds from the same role.
        self.hosts = dict(config.hosts)
        self.ideation_role = resolve_role(self.hosts, self.params, "ideation")
        self.implementation_role = resolve_role(self.hosts, self.params, "implementation")
        self.judge_role = resolve_role(self.hosts, self.params, "judge")
        ideation = self.params["ideation"]
        implementation = self.params["implementation"]
        # Env vars stripped from every session this strategy spawns
        # (ideation, ensemble members, selector, implementation). Used for
        # credential containment: the orchestrating process may hold a key
        # (e.g. OPENAI_API_KEY for the utility LLM) that agent sessions must
        # not inherit. The codex ideation runner strips its own env.
        self.env_strip = list(self.params["env_strip"])
        # Env defaults for every session (set-if-absent in the child env;
        # ambient wrapper values keep precedence). Carries the Bash-tool
        # clock policy so blocking evaluations are possible (finding 14).
        self.env_defaults = dict(self.params["session_env_defaults"])
        # Knowledge-bank serving (serving-agentic-redesign.md): the KAPSO_*
        # env mapping the bank gate resolves on, staged by the campaign
        # launcher (prepare_campaign_serving). None = serving off. Threads
        # into ideation, implementation, and lens-planner sessions — never
        # the feedback judge.
        self.bank_serving = self.params.get("bank_serving")
        # The experience intro's path (wp3-experience.md): what the
        # campaign is served, compiled from the loaded sources and staged
        # into the serving directory AFTER construction (a seeded
        # workspace must be empty then), so it is read when a prompt is
        # built, never here. None = nothing loaded.
        self.experience_intro_path = self.params.get("experience_intro_path")
        # The loaded index's record for the wiki-search gates (an ingest
        # writes it; the facade threads it per campaign).
        self.kg_index_path = self.params.get("kg_index_path")
        # Durable-archive recovery root for the registered evaluation (glob
        # of run archive parents, e.g. "tmp/relbench/*/runs"). None disables
        # archive recovery; the live-process wait still applies.
        self.registered_evaluation_archive_glob = self.params["registered_evaluation_archive_glob"]
        # Ensemble ideation: parallel members, each a spec of its own over
        # the hosts table, pooled by a selector. None: the ideation role
        # alone, one session.
        self.ideation_ensemble = resolve_members(self.hosts, self.params, ideation["ensemble"])
        self.ideation_selector = resolve_optional(self.hosts, self.params, ideation["selector"], name="selector")
        if self.ideation_selector is not None and self.ideation_selector.cli not in SELECTOR_CLIS:
            raise ValueError(
                "search_strategy.params.ideation.selector.cli must be one of "
                f"{', '.join(SELECTOR_CLIS)} (the selector reads the worktree to verify candidates)"
            )
        self.ideation_candidates_per_member = int(ideation["candidates_per_member"])
        if self.ideation_candidates_per_member < 1:
            raise ValueError(
                "search_strategy.params.ideation.candidates_per_member must be >= 1, got "
                f"{self.ideation_candidates_per_member}"
            )
        # A pool of more than one candidate needs a selector.
        if (len(self.ideation_ensemble or [None]) * self.ideation_candidates_per_member > 1
                and self.ideation_selector is None):
            raise ValueError(
                "search_strategy.params.ideation: more than one candidate per iteration "
                "(members × candidates_per_member) needs a selector"
            )
        # Optional member/selector ideation time split (design #5). Default
        # None = NO split: each role gets the full clamped ideation timeout.
        self.ensemble_time_split = normalize_ensemble_time_split(ideation["ensemble_time_split"])
        # Optional task-aware lens planning: a web-enabled Claude session
        # designs the member lenses for THIS task at iteration 1, then a
        # keep-or-revise session re-judges the plan against the campaign
        # evidence EVERY later iteration (current plan in
        # .kapso/lens_plan.json, audit trail in lens_plan_history.jsonl).
        # Static member lenses are forbidden while it is enabled.
        self.ideation_lens_planner = resolve_optional(
            self.hosts, self.params, ideation["lens_planner"], name="lens_planner",
        )
        validate_lens_planner(self.ideation_lens_planner, self.ideation_ensemble)
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
        # The hosted gate is an experience source: loading it mounts it
        # (below), and a config list that names it does not — what is
        # loaded is what is served, and the records name it.
        self.hosted = self.params.get("hosted")
        self.ideation_gates = [gate for gate in ideation["gates"] if gate != HOSTED_GATE]
        # A crashed implementation session (provider safety classifier, CLI
        # abort) leaves the lane's remaining time unused. When set, a crash
        # retries ONCE on this model — contest 5 (2026-08-06) lost both lanes
        # to codex's "flagged for possible cybersecurity risk" kill on an
        # adversarial-robustness task.
        self.implementation_fallback_model = implementation["fallback_model"]
        self.gate_failure_policy = self.params["gate_failure_policy"]
        self.implementation_gates = [gate for gate in implementation["gates"] if gate != HOSTED_GATE]
        # The campaign inbox (docs/research/evolve-hub-design.md v4): with
        # it on, implementation sessions get the request_from_user gate and
        # a session that asks is stopped; the campaign pauses until the
        # person replies. One implementation lane only for now.
        self.inbox_settings = resolve_inbox_settings(
            self.params.get("inbox"),
            self.node_expansion_value,
            self.implementation_gates,
            self.workspace_dir,
        )
        # A staged bank MUST be reachable: serving injects an intro that
        # instructs sessions to call bank_index / bank_get_card /
        # bank_get_card_with_evidence, so the gate that provides them is
        # not an independent config choice — it follows the serving
        # decision. Without this, the intro advertised tools no session
        # had and every pull log stayed empty (E2E review 2026-08-24,
        # blocker 1).
        if self.bank_serving:
            for gates in (self.ideation_gates, self.implementation_gates):
                if "bank" not in gates:
                    gates.append("bank")
        # Same law for the knowledge store: when an index is loaded the
        # wiki-search gates must be mounted, or an ingest writes pages
        # no campaign can read (E2E review 2026-08-24, blocker 2).
        if self.params.get("kg_index_path"):
            for gates in (self.ideation_gates, self.implementation_gates):
                for gate in ("idea", "code"):
                    if gate not in gates:
                        gates.append(gate)
        # And for the hosted source: loading it is what mounts its server
        # (docs/plans/redesign/wp3-experience.md); no config list names it.
        if self.hosted:
            for gates in (self.ideation_gates, self.implementation_gates):
                gates.append(HOSTED_GATE)
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
                self.workspace_dir, self.params["shared_cache_dir"]
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
        print(f"  - ideation: {self.ideation_role.label}")
        print(f"  - implementation: {self.implementation_role.label}")
        print(f"  - judge: {self.judge_role.label}")
        print(f"  - ideation_gates: {self.ideation_gates}")
        print(f"  - implementation_gates: {self.implementation_gates}")
        print(f"  - gate_failure_policy: {self.gate_failure_policy}")
        # A gate the user has not set up is silently absent for the whole
        # campaign; say so once, with the fix, where the gate lists are.
        configured_gates = list(dict.fromkeys(
            list(self.ideation_gates) + list(self.implementation_gates)
        ))
        for gate_name, fix, missing in user_setup_gaps(configured_gates):
            print(
                f"  - gate '{gate_name}' is OFF this campaign "
                f"(missing {', '.join(missing)}): {fix}"
            )
        print(f"  - parent_policy: {self.parent_policy}")
        if self.parent_policy == "weighted":
            print(f"  - parent_sampling: {self.parent_sampling}")
        print(f"  - experiment_history_path: {self.experiment_history_path}")
        print(f"  - feedback_generator: {'configured' if self.feedback_generator else 'not configured'}")
        

    def run(self, context: Any, budget_progress: float = 0.0) -> SearchNode:
        """
        Execute one iteration of generic search.
        
        Node lifecycle:
        1. Generate solution (agent queries experiment history via MCP)
        2. Implement (developer agent handles implementation + evaluation)
        3. Extract results from agent output
        4. Generate feedback
        
        A node that asked the person and whose requests are all answered
        is continued instead (the inbox continuation): no new ideation,
        the same node and branch, the stored CLI session resumed with the
        reply. A suspended node whose request is still open raises — the
        orchestrator never calls run() in that state.
        
        Args:
            context: Either a ContextData object (legacy) or a problem string
            budget_progress: Budget progress percentage (0-100)
        
        Returns:
            SearchNode with solution, evaluation_output, feedback, should_stop
        """
        # Extract problem from context (support both string and ContextData)
        if isinstance(context, str):
            problem = context
        else:
            problem = str(getattr(context, "problem", context))

        self.finishing_cut = None
        waiting = self._answered_suspended_node()
        if waiting is not None:
            # A continuation is not a new round: the last round's duration
            # stands for the next admission.
            return self._continue_node(waiting, problem)

        self.iteration_count += 1
        print(f"\n[GenericSearch] Iteration {self.iteration_count}, budget: {budget_progress:.1f}%")

        # An eval-only VALIDATE grant short-circuits the whole lifecycle:
        # no ideation, no implementation — one full-fidelity measurement of
        # an existing artifact, appended to its node.
        decision = self.fidelity_decision
        if decision is not None and decision.profile == PROFILE_VALIDATE:
            return self._run_validate(decision)
        
        iteration_started_monotonic = time.monotonic()
        iteration_started_at = datetime.now(timezone.utc).isoformat()

        # Select the branch and its node ID once so the recorded lineage, the
        # ideation view, and the implementation base cannot diverge.
        parent = self._select_parent()

        # Step 1: Generate solution(s). With node_expansion_value > 1 the
        # selector emits a ranked top-K; each entry becomes one lane.
        self._status_phase("ideation")
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

        representative = self._expand_round(
            problem=problem,
            solutions=solutions,
            parent=parent,
            decision=decision,
            ideation_sections=ideation_sections,
            ideation_telemetry=ideation_telemetry,
            iteration_started_at=iteration_started_at,
            iteration_started_monotonic=iteration_started_monotonic,
        )
        # The round's wall time, from its ideation's start to its last
        # lane's judgment: what the next admission must cover.
        self.last_round_seconds = time.monotonic() - iteration_started_monotonic
        return representative

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
            iteration=self.iteration_count,
        )
        node.started_at = iteration_started_at
        node.phase_telemetry["ideation"] = ideation_telemetry
        self.in_flight_nodes.append(node)
        if ideation_telemetry:
            self.ideation_bills = []
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

        self._status_phase("implementation")
        (
            agent_output, implementation_telemetry, recovered, suspended, build_error,
        ) = self._implement(
            solution=solution,
            problem=problem,
            branch_name=branch_name,
            parent_branch_name=parent.branch_name,
            ideation_repo_memory_sections_consulted=ideation_sections,
            lane_index=lane_index,
            node_id=node.node_id,
        )
        node.phase_telemetry["implementation"] = implementation_telemetry

        # Update node with implementation results
        node.branch_name = branch_name
        node.parent_branch_name = parent.branch_name
        node.agent_output = agent_output
        if not implementation_telemetry and build_error is not None:
            # The session was not launched (no room under the floor): no
            # branch exists; the node records why and is judged by nobody.
            node.had_error = True
            node.error_message = build_error
            node.evaluation_valid = False
            node.score = None
            node.unjudged = f"the implementation was not launched: {build_error}"
            return node
        node.code_diff = self._get_code_diff(branch_name, parent.branch_name)
        self._absorb_implementation(
            node, agent_output, recovered, suspended, build_error, lane_tag
        )
        return node

    def _absorb_implementation(
        self,
        node: SearchNode,
        agent_output: str,
        recovered: Optional[str],
        suspended,
        build_error: Optional[str],
        lane_tag: str = "",
    ) -> None:
        """Steps 3–3c of the lane: the session's tags into the node, or the
        suspension when the session asked the person and was stopped.
        ``build_error`` is the session's error when it died without
        finishing its report, None when it finished."""
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

        if self._cost_enabled():
            # A build-only session: its score and cost claims are void and
            # nothing is valid until a frame run says so. Its output stays
            # on the node as the judge's evidence for a build that errored;
            # the frame run replaces it when one happens. A session that
            # died is an errored build: nothing to estimate or run. (In the
            # default flow the session also evaluates, and the judge decides
            # whether a dead session left a valid measurement.)
            node.score = None
            node.measured_cost = None
            node.evaluation_valid = False
            node.had_error = build_error is not None
            node.error_message = "" if build_error is None else build_error

        if suspended is not None:
            # The session asked the person and was stopped: nothing to
            # judge, nothing to reconstruct — the node waits for the reply.
            node.suspended = True
            node.request_ids = list(suspended.request_ids)
            node.cli_session_id = suspended.cli_session_id
            print(
                f"{lane_tag}[GenericSearch] node {node.node_id} stopped — asked "
                "the person (requests "
                f"{', '.join(f'#{i}' for i in node.request_ids)}); waiting for "
                "a reply"
            )
            return

        # Step 3b: the implementor is the primary author of
        # technical_difficulties; the fallback reconstructs it when the tag
        # is missing. Purely mechanical trigger — never score/outcome-based.
        self._ensure_technical_difficulties(node)

        # Step 3c: inject the manifest recovered from the durable run archive
        # (returned by this lane's own _implement) so the score of record
        # survives a session that died before printing it.
        if not self._cost_enabled() and recovered and self._manifest_score_of_record(node) is None:
            node.evaluation_output = (
                (node.evaluation_output or "") + "\n" + recovered
            )

    def _answered_suspended_node(self) -> Optional[SearchNode]:
        """The suspended node whose requests the person has all answered,
        or None when nothing is suspended. A suspended node still waiting
        raises: the orchestrator pauses the campaign in that state and
        never runs an iteration."""
        settings = getattr(self, "inbox_settings", None)
        if not settings or not settings.get("enabled"):
            return None
        suspended = [
            node for node in self.node_history if getattr(node, "suspended", False)
        ]
        if not suspended:
            return None
        node = min(suspended, key=lambda item: item.node_id)
        requests = load_requests(settings["path"])
        if not all_answered(requests, node.request_ids):
            raise InboxOpenError(
                f"node {node.node_id} waits for a reply to requests "
                f"{', '.join(f'#{i}' for i in node.request_ids)}"
            )
        return node

    def continues_an_answered_node(self) -> bool:
        return self._answered_suspended_node() is not None

    def _continue_node(self, node: SearchNode, problem: str) -> SearchNode:
        """The inbox continuation: resume the node's stored CLI session with
        the replies, then judge it like any finished node — the same node
        id, the same branch, no new ideation, no iteration counted."""
        settings = self.inbox_settings
        requests = requests_for_ids(load_requests(settings["path"]), node.request_ids)
        follow_up = render_follow_up(requests)
        print(
            f"\n[GenericSearch] Continuing node {node.node_id} on "
            f"{node.branch_name} with the replies to requests "
            f"{', '.join(f'#{i}' for i in node.request_ids)}"
        )
        record_continued(
            settings["path"], node.request_ids,
            node=node.node_id, session=node.cli_session_id,
        )
        continued_at_monotonic = time.monotonic()
        self._status_phase("implementation")
        # The session's bill reaches the node through note_session_spend.
        agent_output, _, recovered, suspended, build_error = self._implement(
            solution=node.solution,
            problem=problem,
            branch_name=node.branch_name,
            parent_branch_name=node.parent_branch_name,
            ideation_repo_memory_sections_consulted=None,
            lane_index=0,
            node_id=node.node_id,
            continuation=Continuation(node.cli_session_id, follow_up),
        )
        node.agent_output = agent_output
        node.code_diff = self._get_code_diff(node.branch_name, node.parent_branch_name)
        node.suspended = False
        node.request_ids = []
        self._absorb_implementation(
            node, agent_output, recovered, suspended, build_error
        )
        if node.suspended:
            # Asked again: the orchestrator pauses the campaign again.
            return node
        self._finalize_nodes([node], continued_at_monotonic, append=False)
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
        self.in_flight_nodes = []
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
                            ideation_sections,
                            # One session fed every lane and it is billed
                            # once: the ledger, node.cost_usd and the campaign
                            # knowledge totals all sum phases across nodes.
                            ideation_telemetry if args[2] == 0 else {},
                            iteration_started_at, args[2],
                        ),
                        lane_args,
                    )
                )

        self._finalize_nodes(nodes, iteration_started_monotonic, append=True)

        representative = pick_representative(
            nodes, self.problem_handler.maximize_scoring,
            getattr(self, "cost_config", None),
        )
        if lane_count > 1:
            representative.should_stop = any(n.should_stop for n in nodes)
            print(
                "[GenericSearch] Round winner: node "
                f"{representative.node_id} (score={representative.score}); "
                f"stop={representative.should_stop}"
            )
        return representative

    def _finalize_nodes(
        self,
        nodes: List[SearchNode],
        iteration_started_monotonic: float,
        append: bool,
    ) -> None:
        """Post-barrier: integrity + feedback serialized in node-id order —
        deterministic history, no interleaved feedback sessions. A node
        that asked the person is stamped and recorded, never judged: it
        is not finished. In cost mode a build refused by the cost check is
        recorded with its refusal as feedback, never judged; every other
        finalized node is judged."""
        self._status_phase("feedback")
        for lane_index, node in enumerate(nodes):
            if getattr(node, "suspended", False):
                print(
                    f"[GenericSearch] node {node.node_id} waits for the person "
                    "— not judged until it is continued"
                )
            elif node.unjudged:
                # A node whose session was never launched: nothing to
                # evaluate, verify or judge; the reason is on the node.
                print(f"[GenericSearch] node {node.node_id} {node.unjudged}")
            elif self.enforce_evaluation_integrity(node):
                if self._evaluate_admitted_candidate(node, lane_index=lane_index):
                    node.commit = self._attempt_commit(node.branch_name)
                    self._verify_attempt(node)
                    # The judge's phase, published again after a verification
                    # (which publishes its own).
                    self._status_phase("feedback")
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
            if append:
                self.in_flight_nodes = [
                    other for other in self.in_flight_nodes if other is not node
                ]
                self.node_history.append(node)
            if not getattr(node, "suspended", False):
                print(
                    f"[GenericSearch] ✓ Node {node.node_id} completed: "
                    f"score={node.score}, should_stop={node.should_stop}"
                )

    def _verify_attempt(self, node: SearchNode) -> None:
        """The harness's verification session for one attempt, between the
        evaluation and the judge, when the harness has verification
        content: the attempt's diff and evaluation output at named paths,
        a detached worktree of its branch, and a verdict file the session
        writes. The verdict is the node's own state: the judge reads it
        and a void is ANDed into validity after the judge; delivery skips
        an unverified attempt. A session that writes no verdict runs once
        more, then the attempt is unverified."""
        harness = self.harness
        if harness is None or not harness.has_content("verification"):
            return
        self._status_phase("verification")
        # The attempt's own host verifies it: the implementation CLI, so a
        # campaign on Codex is verified under Codex's hooks and roles.
        host = self.implementation_role.cli
        directory = verification_dir(Path(self.workspace_dir), node.branch_name)
        directory.mkdir(parents=True, exist_ok=True)
        # The attempt's own changes: the engine's bookkeeping under .kapso/
        # (repo memory, for one) is committed on the branch too and is not
        # the verifier's to read.
        (directory / "diff.patch").write_text(
            self.workspace.repo.git.diff(f"main...{node.branch_name}", "--", ".", ":(exclude).kapso"),
            encoding="utf-8",
        )
        (directory / "evaluation_output.txt").write_text(node.evaluation_output or "", encoding="utf-8")
        verdict_path = directory / VERDICT_FILE
        if verdict_path.exists():
            verdict_path.unlink()
        started = time.monotonic()
        cost = 0.0
        failure = None
        _deadline, not_launched = self._finishing_deadline(
            "verification", float(harness.settings["verification_minutes"]) * 60.0,
        )
        if not_launched is not None:
            # A finishing session with less room than the floor is not
            # launched; the attempt says why, and the round reports the cut.
            node.verification = unverified(f"the verification session was not launched: {not_launched}")
            node.phase_telemetry["verification"] = {"cost_usd": 0.0, "duration_seconds": 0.0}
            self._note_finishing_cut(f"the verification of node {node.node_id} was not launched: {not_launched}")
            print(f"[GenericSearch] Verification of node {node.node_id}: {node.verification['summary']}")
            return
        with self.workspace.materialize_ref(node.branch_name) as worktree:
            staged = self.stage_harness_for("verification", worktree, f"verification/{node.branch_name}", host)
            prompt = render_prompt(load_prompt(VERIFICATION_PROMPT), {
                "branch": node.branch_name,
                "diff_path": str(directory / "diff.patch"),
                "evaluation_output_path": str(directory / "evaluation_output.txt"),
                "verdict_path": str(verdict_path),
                "task_statement_path": str(Path(worktree) / TASK_STATEMENT_FILE),
                "domain_guidance": optional_block(staged.guidance),
            })
            own = {
                "env_strip": self.env_strip,
                "env_defaults": self.env_defaults,
                "allowed_tools": ["Read", "Bash", "Write", *staged.allowed_tools],
                "streaming": True,
                # The verdict lives outside the worktree; codex's sandbox
                # must be told it may write there.
                **({"add_dirs": [str(directory)]} if host == "codex" else {}),
            }
            for _attempt in range(int(harness.settings["verification_attempts"])):
                # The room at each attempt: a retry gets what is left, not
                # what the first attempt was given, and none under the floor.
                deadline, not_launched = self._finishing_deadline(
                    "verification", float(harness.settings["verification_minutes"]) * 60.0,
                )
                if not_launched is not None:
                    failure = failure or f"no room for another attempt: {not_launched}"
                    self._note_finishing_cut(f"the verification of node {node.node_id} was cut: {failure}")
                    break
                own["timeout"] = deadline
                config = CodingAgentFactory.session_config(self.implementation_role, own, staged)
                agent = CodingAgentFactory.create(config)
                agent.initialize(worktree)
                result = agent.generate_code(prompt)
                cost += agent.get_cumulative_cost()
                agent.cleanup()
                if verdict_path.is_file():
                    break
                failure = result.error if not result.success else None
                killed = bool(result.metadata.get("deadline_exceeded")) and not result.metadata.get("completed_before_kill")
                if killed and own["timeout"] < float(harness.settings["verification_minutes"]) * 60.0:
                    # The budget, not the session's own cap, set the deadline
                    # that cut it: a retry would get less still.
                    failure = failure or "the session was cut at the budget's deadline without a verdict"
                    self._note_finishing_cut(f"the verification of node {node.node_id} was cut: {failure}")
                    break
        node.verification = read_verdict(verdict_path)
        if node.verification["verdict"] == "unverified" and failure:
            # Why nothing was written, for the judge and the record: a
            # session that died is not one that chose silence.
            node.verification["summary"] = f"the verification session failed: {failure}"
        node.phase_telemetry["verification"] = {
            "cost_usd": cost, "duration_seconds": time.monotonic() - started,
        }
        print(
            f"[GenericSearch] Verification of node {node.node_id}: "
            f"{node.verification['verdict']} — {node.verification['summary']}"
        )

    def _cost_enabled(self) -> bool:
        return bool(getattr(self, "cost_config", {}).get("enabled"))

    def _evaluate_admitted_candidate(self, node: SearchNode, *, lane_index: int = 0) -> bool:
        """Cost mode: estimate a finalized build and frame-run it when
        admitted. Returns whether the judge reads the node, which is every
        node except a refused one: an errored build gets no estimate and
        no run and is judged on its session output; a failed run is judged
        on what it captured."""
        if not self._cost_enabled() or node.had_error:
            return True
        if self.before_candidate_evaluation is None:
            raise ValueError("Cost-enabled search requires before_candidate_evaluation")
        if not self.before_candidate_evaluation(node):
            return False
        self.evaluate_candidate(node, lane_index=lane_index)
        return True

    def evaluate_candidate(self, node: SearchNode, *, lane_index: int = 0) -> SearchNode:
        """Execute a finalized candidate after its cost estimate was admitted.

        The run's capture replaces the session's output as the judge's
        evidence; the session's manifest line and score never count.
        """
        if not self._cost_enabled() or node.estimated_cost is None or node.cost_refusal:
            raise ValueError("Deferred evaluation requires an admitted cost estimate")
        validate_cost_value(node.estimated_cost, self.cost_config["unit"])
        self._status_phase("evaluation")
        started = time.monotonic()
        deadline, not_launched = self._finishing_deadline("evaluation", self.implementation_role.timeout)
        if not_launched is not None:
            node.evaluation_output = f"Evaluation not run: {not_launched}"
            node.evaluation_valid = False
            node.phase_telemetry["evaluation"] = {"duration_seconds": 0.0}
            self._note_finishing_cut(f"the evaluation run of node {node.node_id} was not launched: {not_launched}")
            print(f"[GenericSearch] node {node.node_id} {node.evaluation_output}")
            return node
        run = execute_candidate_evaluation(
            node,
            deadline_seconds=deadline,
            registered_evaluation_command=self.registered_evaluation_command,
            workspace=self.workspace, workspace_dir=self.workspace_dir,
            env_strip=self.env_strip,
            env_defaults=self.env_defaults,
            env_overrides=lane_env_overlay(self.expansion_lane_env, lane_index),
        )
        node.phase_telemetry["evaluation"] = {
            "duration_seconds": time.monotonic() - started,
        }
        node.measured_cost = measured_cost_from_output(
            node.evaluation_output, self.cost_config["unit"],
        )
        # The failed and overran lines are the runner's; this is the
        # success line, with what the run measured or that it did not.
        if run.completed:
            measured = (
                f"measured {node.measured_cost['amount']} {node.measured_cost['unit']}"
                if node.measured_cost is not None else
                f"no measured cost; estimate {node.estimated_cost['amount']} "
                f"{node.estimated_cost['unit']} stands"
            )
            print(
                f"[GenericSearch] node {node.node_id} evaluation ran in "
                f"{run.duration_seconds:.1f}s; {measured}"
            )
        return node

    def _note_ideation_spend(self, cost_usd: float) -> None:
        """An ideation session just ended. Its round has no node to carry the
        bill until every session has finished, and those run for a long time
        and can raise. Members run on threads, hence the list."""
        self.ideation_bills.append(cost_usd)

    def _generate_solution(
        self, problem: str, parent_branch: str
    ) -> Tuple[List[str], List[str], Dict[str, float]]:
        """Generate solution(s) using Claude Code with MCP gates; returns
        (solutions, sections_consulted, phase_telemetry)."""
        self.ideation_bills = []
        return generate_solution(
            problem,
            parent_branch,
            workspace=self.workspace,
            llm=self.llm,
            experiment_history_path=self.experiment_history_path,
            ideation_gates=self.ideation_gates,
            gate_failure_policy=self.gate_failure_policy,
            bank_serving=self.bank_serving,
            kg_index_path=self.kg_index_path,
            hosted=self.hosted,
            web=self.ideation_role.web,
            run_ensemble=self._generate_solution_ensemble,
            session=f"ideation/iter{self.iteration_count}",
        )

    def _run_lens_planner_session(self, prompt: str, ideation_dir: str):
        """One planner/replanner claude session; returns (result, cost_usd)."""
        self._status_phase("lens_planning")
        result = run_lens_planner_session(
            prompt,
            ideation_dir,
            planner=self.ideation_lens_planner,
            bank_serving=self.bank_serving,
            env_strip=self.env_strip,
            env_defaults=self.env_defaults,
            clamped_timeout=self._clamped_timeout,
            artifacts_dir=self._ideation_artifacts_dir(),
            staged=self.stage_harness_for("ideation", ideation_dir, "lens_planner", self.ideation_lens_planner.cli),
        )
        self._note_ideation_spend(result[1])
        # Planning runs inside the ideation step; flip the live phase back
        # so the sessions that follow are not reported as still planning.
        self._status_phase("ideation")
        return result

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
            node_history=self.node_history,
            campaign_state_brief=self._campaign_state_brief,
            get_best_experiment=self.get_best_experiment,
            run_planner_session=self._run_lens_planner_session,
            domain_guidance=self.stage_harness_for(
                "ideation", ideation_dir, "lens_planner", self.ideation_lens_planner.cli,
            ).guidance,
            experience_intro=self.experience_intro(),
        )

    def _generate_solution_ensemble(
        self,
        problem: str,
        repo_memory_brief: str,
        ideation_dir: str,
        mcp_servers: Dict[str, Any],
        ideation_allowed_tools: List[str],
    ) -> Tuple[str, List[str], Dict[str, float]]:
        """Fan out ideation across CLI members, then select solution(s)."""
        return generate_solution_ensemble(
            problem=problem,
            repo_memory_brief=repo_memory_brief,
            ideation_dir=ideation_dir,
            mcp_servers=mcp_servers,
            ideation_allowed_tools=ideation_allowed_tools,
            members=self.ideation_ensemble or [self.ideation_role],
            ideation_candidates_per_member=self.ideation_candidates_per_member,
            ensemble_time_split=self.ensemble_time_split,
            env_strip=self.env_strip,
            env_defaults=self.env_defaults,
            clamped_timeout=self._clamped_timeout,
            ideation_timeout=self.ideation_role.timeout,
            artifacts_dir=self._ideation_artifacts_dir(),
            build_prompt=self._build_ideation_prompt,
            resolve_lenses=self._resolve_member_lenses,
            select_candidates=self._select_from_candidates,
            note_session_spend=self._note_ideation_spend,
            stage_harness_for=self.stage_harness_for,
        )

    def _campaign_state_brief(self) -> str:
        """Factual campaign trajectory for the selector's return judgment."""
        return campaign_state_brief(
            self.node_history, self.problem_handler.maximize_scoring
        )

    def _select_from_candidates(
        self,
        problem: str,
        repo_memory_brief: str,
        pool: List[Dict[str, str]],
        ideation_dir: str,
        selector_deadline: float,
    ) -> Dict[str, Any]:
        """Run the selector-critic session over the pooled candidates."""
        return select_from_candidates(
            problem=problem,
            repo_memory_brief=repo_memory_brief,
            pool=pool,
            ideation_dir=ideation_dir,
            selector_deadline=selector_deadline,
            ideation_selector=self.ideation_selector,
            node_expansion_value=self.node_expansion_value,
            campaign_state=self._campaign_state_brief(),
            env_strip=self.env_strip,
            env_defaults=self.env_defaults,
            artifacts_dir=self._ideation_artifacts_dir(),
            note_session_spend=self._note_ideation_spend,
            stage_harness_for=self.stage_harness_for,
        )

    def _build_ideation_prompt(
        self,
        problem: str,
        repo_memory_brief: str,
        allowed_tools: Sequence[str] = (),
        domain_guidance: str = "",
    ) -> str:
        """Build the ideation prompt for Claude Code; ``allowed_tools`` is
        the session's whitelist, which decides whose gate guidance the
        prompt carries."""
        return build_ideation_prompt(
            problem,
            repo_memory_brief,
            budget_status=self._render_budget_status(),
            shared_artifacts_brief=self.shared_artifacts_brief,
            inbox_ideation=self._render_inbox_ideation(),
            knowledge_tools=knowledge_tools_block("ideation", allowed_tools),
            domain_guidance=domain_guidance,
            experience_intro=self.experience_intro(),
        )

    def experience_intro(self) -> str:
        """The intro the campaign's sessions read — what it is served,
        measured before — from the serving directory the launcher staged;
        "" when nothing is loaded. Read at prompt time: a path with no file
        behind it is a staging failure, and raises."""
        if not self.experience_intro_path:
            return ""
        return Path(self.experience_intro_path).read_text()

    def _render_inbox_ideation(self) -> str:
        """The inbox block for ideation: the rule against designing around
        a person-only gap, and what the person already answered (empty
        with the inbox off)."""
        settings = getattr(self, "inbox_settings", None)
        if not settings or not settings.get("enabled"):
            return ""
        return render_inbox_ideation(load_requests(settings["path"]))

    def waiting_requests(self) -> List[Request]:
        """Open requests of the nodes that asked the person (design v4):
        while any exist the campaign pauses instead of iterating."""
        settings = getattr(self, "inbox_settings", None)
        if not settings or not settings.get("enabled"):
            return []
        waiting_nodes = {
            node.node_id
            for node in self.node_history
            if getattr(node, "suspended", False)
        }
        if not waiting_nodes:
            return []
        return [
            request
            for request in load_requests(settings["path"]).values()
            if request.open and request.node in waiting_nodes
        ]

    def _salvage_ideation_output(self, result) -> Optional[str]:
        """Recover a deadline-terminated ideation's partial output
        (None for crashes and near-empty kills)."""
        return salvage_ideation_output(result)

    def _implement(
        self,
        solution: str,
        problem: str,
        branch_name: str,
        parent_branch_name: str = "main",
        ideation_repo_memory_sections_consulted: Optional[List[str]] = None,
        lane_index: int = 0,
        node_id: int = 0,
        continuation: Optional[Continuation] = None,
    ) -> Tuple[str, Dict[str, float], Optional[str], Any, Optional[str]]:
        """Implementation session with MCP gates; returns (agent output,
        phase telemetry, recovered manifest line, suspended-session record
        or None, the session's error when it died or None)."""
        lane_env = lane_env_overlay(self.expansion_lane_env, lane_index)
        room = self._room()
        if room is not None and room < self.budget_snapshot.min_agent_timeout_seconds:
            # A search session with less room than the floor is not
            # launched: the lane records why, and the next admission reads
            # the round that overran.
            reason = (
                f"the budget left {max(room, 0.0):.0f}s for the implementation, "
                f"under the {self.budget_snapshot.min_agent_timeout_seconds:.0f}s floor"
            )
            print(f"[GenericSearch] Implementation of {branch_name} not launched: {reason}")
            return f"Implementation not launched: {reason}", {}, None, None, reason
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
            bank_serving=self.bank_serving,
            kg_index_path=self.kg_index_path,
            hosted=self.hosted,
            implementation_role=self.implementation_role,
            implementation_fallback_model=self.implementation_fallback_model,
            env_strip=self.env_strip,
            env_defaults=self.env_defaults,
            lane_env=lane_env,
            clamped_timeout=self._clamped_timeout,
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
                lambda facts: self._session_end_facts.update({node_id: facts})
            ),
            note_session_spend=(
                lambda telemetry: self._note_session_spend(
                    node_id, telemetry, continued=continuation is not None
                )
            ),
            await_registered_evaluation=(
                lambda output: self._await_registered_evaluation(output, branch_name)
            ),
            node_id=node_id,
            inbox_settings=getattr(self, "inbox_settings", None),
            continuation=continuation,
            defer_evaluation=self._cost_enabled(),
            stage_harness_for=self.stage_harness_for,
        )

    def _note_session_spend(
        self, node_id: int, telemetry: Dict[str, float], *, continued: bool
    ) -> None:
        """A build session just ended and its bill is known, ahead of the
        close and the evaluation wait that can outlive the process. A lane's
        note replaces the last one, since its fallback session reports the
        running total. A continued node is in the history and its sessions
        add up: cost, duration and the per-gate knowledge calls alike."""
        if continued:
            node = self.node_history[node_id]
            prior = node.phase_telemetry.get("implementation", {})
            node.phase_telemetry["implementation"] = {
                key: prior.get(key, 0.0) + telemetry.get(key, 0.0)
                for key in list(prior) + [
                    key for key in telemetry if key not in prior
                ]
            }
            return
        for node in self.in_flight_nodes:
            if node.node_id == node_id:
                node.phase_telemetry["implementation"] = telemetry

    def _build_implementation_prompt(
        self,
        solution: str,
        problem: str,
        branch_name: str,
        repo_memory_brief: str,
        repo_memory_detail_access_instructions: str,
        previous_errors: str,
        lane_brief: str = "",
        inbox_section: str = "",
        allowed_tools: Sequence[str] = (),
        domain_guidance: str = "",
    ) -> str:
        """Build the implementation prompt for Claude Code; ``allowed_tools``
        is the session's whitelist, which decides whose gate guidance the
        prompt carries."""
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
            inbox_section=inbox_section,
            knowledge_tools=knowledge_tools_block("implementation", allowed_tools),
            domain_guidance=domain_guidance,
            experience_intro=self.experience_intro(),
        )

    def _manifest_of_record(self, node: SearchNode) -> Optional[Dict[str, Any]]:
        """The granted-class manifest from the last manifest line of the
        evaluation that ran: the session's in the default flow, Kapso's
        frame run in cost mode, where a session's own line never counts."""
        if not self.registered_evaluation_command:
            return None
        if self._cost_enabled() and not node.frame_run_outcome:
            return None
        return manifest_of_record(
            node,
            registered_evaluation_command=self.registered_evaluation_command,
            fidelity_decision=self.fidelity_decision,
            registered_subsample_seed=self.registered_subsample_seed,
        )

    def _manifest_score_of_record(self, node: SearchNode) -> Optional[float]:
        """The granted-class score from the manifest of record."""
        manifest = self._manifest_of_record(node)
        return None if manifest is None else float(manifest["score"])

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
        self._status_phase("evaluation")
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
        # nodes never reach the bridge (integrity errors are filtered); a
        # harness void stands whatever the measurement.
        node.evaluation_valid = not is_void(node.verification)
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
        """The evaluation contract the implementer follows: build-only in
        cost mode, where Kapso runs the evaluator after admission; otherwise
        the registered contract when a maintainer owns evaluation, the
        provided suite's when the build brought one, build-your-own else."""
        provided = bool(self.provided_evaluation_manifest)
        if self._cost_enabled():
            return deferred_evaluation_instructions(self.registered_evaluation_command, provided=provided)
        return evaluation_instructions(self.registered_evaluation_command, provided=provided)

    def _ensure_technical_difficulties(self, node) -> None:
        """Run the fallback reconstruction when the implementor's
        technical_difficulties tag is missing (purely mechanical trigger —
        never score/outcome-based)."""
        if (node.technical_difficulties or "").strip():
            return
        deadline, not_launched = self._finishing_deadline("reconstruction", None)
        if not_launched is not None:
            node.technical_difficulties = f"(the reconstruction was not launched: {not_launched})"
            self._note_finishing_cut(f"the reconstruction for node {node.node_id} was not launched: {not_launched}")
            return
        # Staged like every session, on the implementation role's host, in
        # the campaign (the branch is read there): its record is
        # `difficulties/<branch>`.
        staged = self.stage_harness_for(
            "implementation", self.workspace_dir, f"difficulties/{node.branch_name}", self.implementation_role.cli,
        )
        ensure_technical_difficulties(
            node,
            implementation_role=self.implementation_role,
            env_strip=self.env_strip,
            deadline_seconds=deadline,
            workspace_dir=self.workspace_dir,
            session_stream_path=self._session_stream_path,
            staged=staged,
        )

    def _await_registered_evaluation(self, output_text: str, session: str):
        """Teardown guard for the registered evaluation: wait for a live
        grader, then recover the manifest line from the durable run archive
        of ``session`` (must run BEFORE finalize_session's rmtree)."""
        return await_registered_evaluation(
            output_text,
            session=session,
            registered_evaluation_command=self.registered_evaluation_command,
            registered_evaluation_archive_glob=(
                self.registered_evaluation_archive_glob
            ),
            clamped_timeout=self._clamped_timeout,
            implementation_timeout=self.implementation_role.timeout,
            session_started_ts=getattr(
                self, "_last_session_started_ts", 0.0
            ),
        )

    def stage_harness_for(self, module: str, session_dir: str, label: str, agent_type: str) -> StagedHarness:
        """What the loaded harness gives one session of a module, staged
        into the session's directory (nothing without a harness, the
        session's record all the same). An ideation session's id is
        prefixed with the iteration, so the staging records sit beside the
        ideation artifacts."""
        session_id = f"ideation/iter{self.iteration_count}/{label}" if module == "ideation" else label
        if self.harness is not None and not session_dir:
            raise ValueError(f"the {module} session {session_id} has no directory to stage into")
        return stage_harness(
            self.harness, module,
            # A session with no directory of its own (the judge's) runs in
            # the campaign; nothing is staged into it without a harness.
            session_dir=Path(session_dir) if session_dir else Path(self.workspace_dir),
            campaign_dir=Path(self.workspace_dir),
            session_id=session_id, host=host_for(agent_type),
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

    # The finishing sessions of an attempt, in order: what must still fit
    # after each is the obligations the later ones carry (the budget's end,
    # docs/plans/redesign/wp4-config.md).
    FINISHING_STAGES = ("reconstruction", "evaluation", "verification", "judge")

    def _room(self, *, finishing: bool = False, obligations_seconds: float = 0.0) -> Optional[float]:
        """The time a session launched now may take; None without a time
        budget. The snapshot is frozen at iteration start; the monotonic
        anchor discounts what this iteration's earlier phases burned."""
        if self.budget_snapshot is None:
            return None
        drift = (
            time.monotonic() - self.budget_snapshot_monotonic
            if self.budget_snapshot_monotonic is not None
            else 0.0
        )
        return self.budget_snapshot.room(
            obligations_seconds, finishing=finishing, elapsed_since_snapshot=drift,
        )

    def _clamped_timeout(self, configured_seconds: Optional[float]) -> Optional[float]:
        """A search session's deadline (setup, ideation, implementation):
        its configured timeout bounded by the searchable remainder, never
        raised; the configured timeout alone without a time budget."""
        if self.budget_snapshot is None:
            return configured_seconds
        return self.budget_snapshot.deadline(configured_seconds, self._room())

    def _obligations_after(self, stage: str) -> float:
        """What must still fit after a finishing ``stage``: the evaluation
        run in cost mode and the verification when the harness verifies,
        each at the floor (a verification cut short is unverified, and the
        judge still speaks); the judge at its reserve — its last observed
        duration, the config's `judge_reserve_seconds` before one, since a
        judge takes minutes, and a judge kept at the floor let a
        verification run the remainder down to 59 s for it, under the
        floor, the hour's attempt then unjudged (the release gate,
        2026-10-30); and the delivery's protected slice."""
        snapshot = self.budget_snapshot
        later = self.FINISHING_STAGES[self.FINISHING_STAGES.index(stage) + 1:]
        reserves = {
            "evaluation": snapshot.min_agent_timeout_seconds if self._cost_enabled() else 0.0,
            "verification": (
                snapshot.min_agent_timeout_seconds
                if self.harness is not None and self.harness.has_content("verification") else 0.0
            ),
            "judge": snapshot.judge_reserve_seconds,
        }
        return sum(reserves[name] for name in later) + snapshot.delivery_seconds

    def _finishing_deadline(
        self, stage: str, configured_seconds: Optional[float],
    ) -> Tuple[Optional[float], Optional[str]]:
        """A finishing session's deadline, or the reason it is not
        launched: its room is the whole remainder less the obligations
        after it, and a room under the floor launches nothing."""
        if self.budget_snapshot is None:
            return configured_seconds, None
        room = self._room(finishing=True, obligations_seconds=self._obligations_after(stage))
        floor = self.budget_snapshot.min_agent_timeout_seconds
        if room is not None and room < floor:
            return None, f"the budget left {max(room, 0.0):.0f}s for the {stage}, under the {floor:.0f}s floor"
        return self.budget_snapshot.deadline(configured_seconds, room), None

    def _note_finishing_cut(self, reason: str) -> None:
        """The first finishing session of the round not launched or cut:
        the orchestrator stops the campaign on it, by name."""
        if self.finishing_cut is None:
            self.finishing_cut = reason

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

        parent = (
            self._sample_parent()
            if self.parent_policy == "weighted"
            else self.get_best_experiment()
        )
        if parent is not None:
            return ParentSelection(
                branch_name=parent.branch_name,
                node_id=parent.node_id,
            )

        # No validly evaluated node exists yet. Committed-but-unevaluated
        # work (a deadline-killed implementation, an evaluation that never
        # ran) is still real progress; branching from `main` strands it on
        # its branch and the next iteration redoes it. Integrity-flagged
        # candidates stay excluded — never build on an evaluator tamperer,
        # nor on an attempt the harness voided.
        committed = [
            node
            for node in self.node_history
            if not node.had_error
            and not node.evaluation_integrity_error
            and not node.cost_refusal
            and not getattr(node, "suspended", False)
            and not is_void(node.verification)
            and not node.unjudged
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

    def _sample_parent(self) -> Optional[SearchNode]:
        """Draw one scored candidate, ranked as `best` ranks them. The draw
        is seeded from the configured seed and the node count, so a resumed
        campaign repeats the choice it would have made uninterrupted."""
        ranked = rank_candidates(self.node_history, self._selection_config())
        if not ranked:
            return None
        sampling = self.parent_sampling
        children = [
            sum(1 for node in self.node_history if node.parent_node_id == parent.node_id)
            for parent in ranked
        ]
        weights = parent_weights(
            children, sampling["temperature"], sampling["offspring_penalty"]
        )
        rng = random.Random(f"{sampling['seed']}:{len(self.node_history)}")
        return rng.choices(ranked, weights=weights)[0]

    def get_experiment_history(self, best_last: bool = False) -> List[SearchNode]:
        """Return all nodes, optionally sorted by score (unscored sort worst)."""
        if best_last:
            return sorted(
                self.node_history,
                key=lambda node: (
                    not node.had_error and node.evaluation_valid and node.score is not None,
                    0.0 if node.score is None
                    else selection_score(node, self._selection_config()),
                )
            )
        return self.node_history

    def get_best_experiment(self) -> Optional[SearchNode]:
        """Return the best successful SCORED node."""
        ranked = rank_candidates(self.node_history, self._selection_config())
        return ranked[0] if ranked else None

    def _selection_config(self) -> Dict[str, Any]:
        return {
            "cost": getattr(self, "cost_config", DEFAULT_COST_CONFIG),
            "maximize_scoring": self.problem_handler.maximize_scoring,
        }

    def get_deliverable_experiment(self) -> Optional[SearchNode]:
        """The node the campaign delivers: the committed-slot winner under
        the registered evaluator (`select_deliverable`), among the attempts
        the harness could verify and the judge read. An attempt the harness
        could not verify stays a parent but is never delivered
        (kapso.harness.verdict); one the judge never read is neither."""
        candidates = [
            node for node in self.node_history
            if is_deliverable(node.verification) and not node.unjudged
        ]
        return select_deliverable(
            candidates,
            evaluator_id=self.registered_evaluator_id,
            selection_config=self._selection_config(),
        )

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

    def _attempt_commit(self, branch_name: str) -> str:
        """The branch's head when the attempt is evaluated: the commit the
        record names for it."""
        return self.workspace.repo.commit(branch_name).hexsha

    def _judge_end_facts(self, node: SearchNode) -> str:
        """Harness ground truth for the judge: how the build session ended
        and, when Kapso frame-ran the evaluator, how that run ended; the
        captured streams never show a deadline kill."""
        session_facts = getattr(self, "_session_end_facts", {}).get(node.node_id, "")
        if not node.frame_run_outcome:
            return session_facts
        run_facts = f"Kapso's evaluation run {node.frame_run_outcome}"
        return f"{session_facts}; {run_facts}" if session_facts else run_facts

    def _generate_feedback(self, node: SearchNode) -> SearchNode:
        """Judge one node with the FeedbackGenerator (updates the node
        in-place with feedback, score, and should_stop); an attempt the
        budget leaves no judge for is `unjudged`, with the reason."""
        configured = (
            self.feedback_generator.configured_timeout_seconds
            if self.feedback_generator is not None else None
        )
        deadline, not_launched = self._finishing_deadline("judge", configured)
        if not_launched is not None:
            node.unjudged = f"the budget left no time for the judge ({not_launched})"
            self._note_finishing_cut(f"node {node.node_id} unjudged: {not_launched}")
            print(f"[GenericSearch] node {node.node_id} unjudged: {node.unjudged}")
            return node
        rooms = [deadline]

        def retry_deadline():
            # The retry's room, re-computed; remembered so the attempt
            # says which deadline cut it.
            room, not_launched = self._finishing_deadline("judge", configured)
            rooms.append(room)
            return room, not_launched

        generate_feedback(
            node,
            feedback_generator=self.feedback_generator,
            goal=self.goal,
            axes=axes_brief(current_plan_axes(self.workspace_dir)),
            session_end_facts=self._judge_end_facts(node),
            deadline_seconds=deadline,
            retry_deadline=retry_deadline,
            manifest_of_record=self._manifest_of_record,
            finalize_run_selection=(
                lambda manifest, valid: label_run_of_record(
                    self.registered_evaluation_archive_glob, manifest, valid
                )
            ),
            # The frame run's verdict: in cost mode a session's own claim
            # of validity never reaches the judge.
            execution_valid=node.evaluation_valid if self._cost_enabled() else True,
            stage_harness_for=self.stage_harness_for,
            verification=node.verification,
        )
        # The deadline that cut the judge was the budget's when the run it
        # cut (the first, or the retry) had a room below the configured
        # timeout: the attempt says so, and the round reports the
        # finishing cut; a retry refused for want of room is one too.
        budget_set = rooms[-1] is not None and (configured is None or rooms[-1] < configured)
        if node.unjudged == FeedbackGenerator.DEADLINE_REASON and budget_set:
            node.unjudged = "the judge's session ended at the budget's deadline"
            self._note_finishing_cut(f"node {node.node_id} unjudged: {node.unjudged}")
        elif node.unjudged.startswith("no tags, and no room for a retry"):
            self._note_finishing_cut(f"node {node.node_id} unjudged: {node.unjudged}")
        return node

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
            "round_seconds": self.last_round_seconds,
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
        self.last_round_seconds = state["round_seconds"]
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
