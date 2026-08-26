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

import os
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
from kapso.execution.search_strategies.generic.ideation import (
    ENSEMBLE_CANDIDATES_PER_MEMBER,
    build_ideation_prompt,
    campaign_state_brief,
    generate_solution,
    generate_solution_ensemble,
    normalize_ensemble_member,
    normalize_ensemble_time_split,
    normalize_ideation_ensemble,
    salvage_ideation_output,
    select_from_candidates,
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


PARENT_POLICIES = frozenset({"best", "baseline"})


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
        - auth_mode: Claude authentication mode: auto, oauth, or api_key
          (default: adapter auto-resolution — API key, then subscription login)
        - ideation_timeout: Ideation session deadline in seconds. Default
          None: no deadline — the session is bounded only by an explicit
          time budget, when one is set.
        - implementation_timeout: Implementation session deadline in
          seconds. Default None: no deadline (budget-bounded only).
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
        - ensemble_time_split: Optional {member_fraction, selector_fraction,
          selector_min_seconds?} carving the ensemble's ideation clamp
          between the member fan-out and the selector. Default absent = NO
          split — each role gets the full clamped ideation timeout, the
          selector's clamp recomputed after the members finish.
        - parent_policy: Parent branch selection: best or baseline (default: best).
          Under `best`, before any validly evaluated node exists, the latest
          committed non-error, non-tampered node is used so in-progress work
          continues in place; `main` only when no committed work exists.
        - ideation_gates: MCP gates for ideation (default: ["research", "experiment_history", "repo_memory", "leeroopedia"])
        - implementation_gates: MCP gates for implementation (default: ["research", "repo_memory", "leeroopedia"])
        - implementation_web: Live-web access in implementation sessions —
          claude WebSearch/WebFetch and codex --search (default: True).
          Independent of the ideation `web_search` knob.
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
            "claude-opus-5"
        )
        # Subscription-first (bedrock removed 2026-08-26 by user
        # direction): an explicit auth_mode threads through; otherwise the
        # adapter auto-resolves (API key, then the CLI subscription login).
        if self.params.get("auth_mode") is not None:
            self._claude_auth_settings = {"auth_mode": self.params["auth_mode"]}
        else:
            self._claude_auth_settings = {}
        # None (the default) means no session deadline: sessions run to
        # completion, bounded only by the time budget when one exists.
        self.ideation_timeout = self.params.get("ideation_timeout")
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
        # Knowledge-bank serving (serving-agentic-redesign.md): the KAPSO_*
        # env mapping the bank gate resolves on, staged by the campaign
        # launcher (prepare_campaign_serving). None = serving off. Threads
        # into ideation, implementation, and lens-planner sessions — never
        # the feedback judge.
        self.bank_serving = self.params.get("bank_serving")
        # Knowledge-store index for the wiki-search gates (learn_knowledge
        # / index_kg write it; the facade threads it per campaign).
        self.kg_index_path = self.params.get("kg_index_path")
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
        # Optional member/selector ideation time split (design #5). Default
        # None = NO split: each role gets the full clamped ideation timeout.
        self.ensemble_time_split = normalize_ensemble_time_split(
            self.params.get("ensemble_time_split")
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
            "claude-opus-5"
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
        # Web access in implementation sessions (independent of the ideation
        # `web_search` knob): gates the claude whitelist's WebSearch/WebFetch
        # AND the codex --search flag. Default True; benchmarks whose
        # protocol forbids live web during implementation (e.g. relbench's
        # temporal-leakage rules) set false in their mode config.
        self.implementation_web = self.params.get("implementation_web", True)
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
        self.implementation_timeout = self.params.get("implementation_timeout")
        self.gate_failure_policy = self.params.get("gate_failure_policy", "warn")
        self.implementation_gates = self.params.get("implementation_gates", ["research", "repo_memory", "leeroopedia"])
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
        # Same law for the knowledge store: when a KG index is staged the
        # wiki-search gates must be mounted, or learn_knowledge writes
        # pages no campaign can read (E2E review 2026-08-24, blocker 2).
        if self.params.get("kg_index_path"):
            for gates in (self.ideation_gates, self.implementation_gates):
                for gate in ("idea", "code"):
                    if gate not in gates:
                        gates.append(gate)
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

        self._status_phase("implementation")
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
        self._status_phase("feedback")
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
        """Generate solution(s) using Claude Code with MCP gates; returns
        (solutions, sections_consulted, phase_telemetry)."""
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
            ideation_web_search=self.ideation_web_search,
            ideation_ensemble=self.ideation_ensemble,
            idea_generation_model=self.idea_generation_model,
            claude_auth_settings=self._claude_auth_settings,
            env_strip=self.env_strip,
            env_defaults=self.env_defaults,
            web_disallowed_tools=self._web_disallowed_tools,
            clamped_timeout=self._clamped_timeout,
            ideation_timeout=self.ideation_timeout,
            session_effort=self.session_effort,
            build_prompt=self._build_ideation_prompt,
            run_ensemble=self._generate_solution_ensemble,
        )

    def _run_lens_planner_session(self, prompt: str, ideation_dir: str):
        """One planner/replanner claude session; returns (result, cost_usd)."""
        self._status_phase("lens_planning")
        result = run_lens_planner_session(
            prompt,
            ideation_dir,
            planner=self.ideation_lens_planner,
            bank_serving=self.bank_serving,
            claude_auth_settings=self._claude_auth_settings,
            env_strip=self.env_strip,
            env_defaults=self.env_defaults,
            web_disallowed_tools=self._web_disallowed_tools,
            ideation_web_search=self.ideation_web_search,
            session_effort=self.session_effort,
            artifacts_dir=self._ideation_artifacts_dir(),
        )
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
        """Fan out ideation across CLI members, then select solution(s)."""
        return generate_solution_ensemble(
            problem=problem,
            repo_memory_brief=repo_memory_brief,
            ideation_dir=ideation_dir,
            mcp_servers=mcp_servers,
            ideation_allowed_tools=ideation_allowed_tools,
            ideation_ensemble=self.ideation_ensemble,
            ideation_candidates_per_member=self.ideation_candidates_per_member,
            ensemble_time_split=self.ensemble_time_split,
            ideation_web_search=self.ideation_web_search,
            claude_auth_settings=self._claude_auth_settings,
            env_strip=self.env_strip,
            env_defaults=self.env_defaults,
            web_disallowed_tools=self._web_disallowed_tools,
            session_effort=self.session_effort,
            clamped_timeout=self._clamped_timeout,
            ideation_timeout=self.ideation_timeout,
            artifacts_dir=self._ideation_artifacts_dir(),
            build_prompt=self._build_ideation_prompt,
            resolve_lenses=self._resolve_member_lenses,
            select_candidates=self._select_from_candidates,
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
            claude_auth_settings=self._claude_auth_settings,
            env_strip=self.env_strip,
            env_defaults=self.env_defaults,
            session_effort=self.session_effort,
            artifacts_dir=self._ideation_artifacts_dir(),
        )

    def _build_ideation_prompt(
        self,
        problem: str,
        repo_memory_brief: str,
    ) -> str:
        """Build the ideation prompt for Claude Code."""
        return build_ideation_prompt(
            problem,
            repo_memory_brief,
            budget_status=self._render_budget_status(),
            shared_artifacts_brief=self.shared_artifacts_brief,
        )

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
            bank_serving=self.bank_serving,
            kg_index_path=self.kg_index_path,
            implementation_cli=self.implementation_cli,
            implementation_model=self.implementation_model,
            implementation_fallback_model=self.implementation_fallback_model,
            implementation_web=self.implementation_web,
            claude_auth_settings=self._claude_auth_settings,
            env_strip=self.env_strip,
            env_defaults=self.env_defaults,
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

    def _clamped_timeout(
        self, configured_seconds: Optional[float]
    ) -> Optional[float]:
        """Bound an agent deadline by the searchable budget, when known.

        ``configured_seconds`` None means the phase has no configured
        deadline: the result is the budget remainder when a budget exists,
        and None (unbounded) otherwise.

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
