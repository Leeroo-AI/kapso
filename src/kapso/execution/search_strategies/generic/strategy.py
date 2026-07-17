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

import json
import logging
import os
import re
import shutil
import signal
import time
from dataclasses import dataclass
from datetime import datetime, timezone
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
from concurrent.futures import ThreadPoolExecutor
from kapso.execution.search_strategies.generic import codex_ideation
from kapso.execution.memories.repo_memory import RepoMemoryManager
from kapso.core.prompt_loader import load_prompt, render_prompt

if TYPE_CHECKING:
    from kapso.execution.search_strategies.generic import FeedbackGenerator

logger = logging.getLogger(__name__)

# Enforcement mechanic (mirrors the coding-agent adapter's deadline grace):
# time granted between SIGTERM and SIGKILL when a frame run overruns.
_FRAME_RUN_KILL_GRACE_SECONDS = 2.0

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
ENSEMBLE_CANDIDATES_PER_MEMBER = 2

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

ENSEMBLE_MEMBER_CLIS = frozenset({"claude_code", "codex"})
_ENSEMBLE_MEMBER_KEYS = frozenset({"cli", "model", "effort", "lens"})


def normalize_ensemble_member(value: Any, role: str) -> Dict[str, str]:
    """Validate one ideation-ensemble member (or selector) config entry."""
    if not isinstance(value, dict):
        raise ValueError(f"{role} must be a mapping, got {type(value).__name__}")
    unknown = sorted(set(value) - _ENSEMBLE_MEMBER_KEYS)
    if unknown:
        raise ValueError(f"{role} has unknown keys: {', '.join(unknown)}")
    cli = value.get("cli")
    if cli not in ENSEMBLE_MEMBER_CLIS:
        allowed = ", ".join(sorted(ENSEMBLE_MEMBER_CLIS))
        raise ValueError(f"{role}.cli must be one of: {allowed}")
    model = value.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"{role}.model must be a non-empty string")
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
          each {cli: claude_code|codex, model, effort?, lens?}; omit for
          single-session ideation (default)
        - ideation_selector: Required with ideation_ensemble — the
          selector-critic session {cli: claude_code, model, effort?}
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
        # Optional reasoning-effort for BOTH agent sessions (ideation and
        # implementation); None keeps the CLI's default.
        self.session_effort = self.params.get("effort")
        # Env vars stripped from every Claude session this strategy spawns
        # (ideation, ensemble members, selector, implementation). Used for
        # credential containment: the orchestrating process may hold a key
        # (e.g. OPENAI_API_KEY for the utility LLM) that agent sessions must
        # not inherit. The codex ideation runner strips its own env.
        self.env_strip = list(self.params.get("env_strip", []))
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
        if self.ideation_selector and self.ideation_selector["cli"] != "claude_code":
            raise ValueError(
                "ideation_selector.cli must be claude_code (the selector "
                "reads the worktree to verify candidates)"
            )
        # Include experiment_history, repo_memory, and leeroopedia gates by default for ideation
        self.ideation_gates = self.params.get("ideation_gates", ["research", "experiment_history", "repo_memory", "leeroopedia"])
        
        # Config params for implementation
        self.implementation_model = self.params.get(
            "implementation_model",
            "us.anthropic.claude-opus-4-5-20251101-v1:0"
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

        # Step 1: Generate solution (agent queries experiment history via MCP)
        solution, ideation_sections, ideation_telemetry = self._generate_solution(
            problem,
            parent.branch_name,
        )
        print(f"[GenericSearch] Generated solution ({len(solution)} chars)")

        # Create node
        node = SearchNode(
            node_id=len(self.node_history),
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
            f"[GenericSearch] Implementing on branch: {branch_name} "
            f"(from {parent.branch_name})"
        )
        
        agent_output, implementation_telemetry = self._implement(
            solution=solution,
            problem=problem,
            branch_name=branch_name,
            parent_branch_name=parent.branch_name,
            ideation_repo_memory_sections_consulted=ideation_sections,
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
            node.evaluation_output = agent_result.get("evaluation_output", agent_output)
            # Score from agent result (may be overridden by feedback generator)
            if agent_result.get("score") is not None:
                node.score = float(agent_result.get("score", 0.0))
            print(f"[GenericSearch] Extracted result from agent JSON")
        else:
            # Fallback: use raw agent output
            node.evaluation_output = agent_output
            print(f"[GenericSearch] Warning: No JSON result from agent, using raw output")
        
        # Step 4: Verify provided evaluation files before accepting any score
        # or feedback derived from them.
        if self.enforce_evaluation_integrity(node):
            self._generate_feedback(node)
            self._record_evaluation_attempt(node)
        else:
            print(
                "[GenericSearch] Rejected invalid provided evaluation: "
                f"{node.evaluation_integrity_error}"
            )
        
        # Stamp iteration totals: wall-clock for the whole iteration, spend as
        # the sum of attributed phase costs.
        node.duration_seconds = time.monotonic() - iteration_started_monotonic
        node.cost_usd = sum(
            phase.get("cost_usd", 0.0)
            for phase in node.phase_telemetry.values()
        )

        # Store node
        self.node_history.append(node)

        print(f"[GenericSearch] ✓ Node {node.node_id} completed: score={node.score}, should_stop={node.should_stop}")

        return node

    def _generate_solution(
        self, problem: str, parent_branch: str
    ) -> Tuple[str, List[str], Dict[str, float]]:
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
            Tuple of (solution_text, sections_consulted, phase_telemetry)
            where phase_telemetry is {"cost_usd": ..., "duration_seconds": ...}
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
                repo_root=ideation_dir,
                include_base_tools=False,
                gate_failure_policy=self.gate_failure_policy,
            )

            # 3. Build restricted tool set (read-only for ideation).
            ideation_allowed_tools = [
                "Read",
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
                    "aws_region": self.aws_region,
                    "mcp_servers": mcp_servers,
                    "allowed_tools": ideation_allowed_tools,
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
                            salvaged,
                            self._extract_sections_consulted(result.output),
                            telemetry,
                        )
                    return self._fallback_solution(problem), [], telemetry

                solution = self._extract_solution_from_output(result.output)
                sections_consulted = self._extract_sections_consulted(
                    result.output
                )

                print(
                    "[GenericSearch] Ideation complete, sections consulted: "
                    f"{sections_consulted}"
                )
                return solution, sections_consulted, telemetry
            finally:
                agent.cleanup()
    
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
        clamp = self._clamped_timeout(self.ideation_timeout)
        member_deadline = max(60.0, clamp * ENSEMBLE_MEMBER_TIME_FRACTION)
        selector_deadline = max(
            ENSEMBLE_SELECTOR_MIN_SECONDS, clamp * ENSEMBLE_SELECTOR_TIME_FRACTION
        )

        base_prompt = self._build_ideation_prompt(
            problem=problem, repo_memory_brief=repo_memory_brief
        )
        addendum_template = load_prompt(
            "execution/search_strategies/generic/prompts/ideation_ensemble_addendum.md"
        )

        def run_member(member: Dict[str, str]) -> Dict[str, Any]:
            prompt = base_prompt + "\n\n" + render_prompt(
                addendum_template,
                {
                    "lens": member.get("lens", "no specific lens — judge freely"),
                    "candidate_count": str(ENSEMBLE_CANDIDATES_PER_MEMBER),
                },
            )
            label = f"{member['cli']}:{member['model']}"
            print(f"[GenericSearch] Ensemble ideation member starting: {label}")
            if member["cli"] == "codex":
                artifacts_dir = os.path.join(
                    self.workspace_dir, ".kapso", "ideation",
                    f"iter{self.iteration_count}",
                )

                def run_codex_once(attempt_deadline: float) -> tuple:
                    return codex_ideation.run_codex_ideation(
                        prompt=prompt,
                        model=member["model"],
                        cwd=ideation_dir,
                        timeout_seconds=attempt_deadline,
                        effort=member.get("effort"),
                        artifacts_dir=artifacts_dir,
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

            config = CodingAgentConfig(
                agent_type="claude_code",
                model=member["model"],
                debug_model=member["model"],
                agent_specific={
                    **self._claude_auth_settings,
                    "env_strip": self.env_strip,
                    "aws_region": self.aws_region,
                    "mcp_servers": mcp_servers,
                    "allowed_tools": ideation_allowed_tools,
                    "timeout": member_deadline,
                    "streaming": True,
                    "planning_mode": False,
                    "effort": member.get("effort", self.session_effort),
                },
            )
            agent = ClaudeCodeCodingAgent(config)
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
        with ThreadPoolExecutor(max_workers=len(members)) as executor:
            member_results = list(executor.map(run_member, members))

        pool: List[Dict[str, str]] = []
        sections: List[str] = []
        total_cost = 0.0
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
                f"candidates={kept}/{ENSEMBLE_CANDIDATES_PER_MEMBER} "
                f"(dropped {dropped}){timing}, "
                f"timed_out={member_result.get('timed_out', False)}, {detail}"
            )
            if kept < ENSEMBLE_CANDIDATES_PER_MEMBER:
                logger.warning(
                    f"[GenericSearch] member {member_result['label']} "
                    f"under-delivered: {kept} of "
                    f"{ENSEMBLE_CANDIDATES_PER_MEMBER} candidates"
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
            return self._fallback_solution(problem), sections, telemetry
        if len(pool) == 1:
            print("[GenericSearch] Single candidate — selector skipped")
            return pool[0]["text"], sections, telemetry

        chosen = self._select_from_candidates(
            problem=problem,
            repo_memory_brief=repo_memory_brief,
            pool=pool,
            ideation_dir=ideation_dir,
            selector_deadline=selector_deadline,
        )
        telemetry["cost_usd"] += chosen["cost_usd"]
        telemetry["duration_seconds"] = time.monotonic() - phase_started
        return chosen["solution"], sections, telemetry

    def _select_from_candidates(
        self,
        problem: str,
        repo_memory_brief: str,
        pool: List[Dict[str, str]],
        ideation_dir: str,
        selector_deadline: float,
    ) -> Dict[str, Any]:
        """Run the selector-critic session over the pooled candidates."""
        from kapso.execution.coding_agents.base import CodingAgentConfig
        from kapso.execution.coding_agents.adapters.claude_code_agent import ClaudeCodeCodingAgent

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
                "candidates": candidates_block,
            },
        )
        selector = self.ideation_selector
        config = CodingAgentConfig(
            agent_type="claude_code",
            model=selector["model"],
            debug_model=selector["model"],
            agent_specific={
                **self._claude_auth_settings,
                "env_strip": self.env_strip,
                "aws_region": self.aws_region,
                "allowed_tools": ["Read"],
                "timeout": selector_deadline,
                "streaming": True,
                "planning_mode": False,
                "effort": selector.get("effort", self.session_effort),
            },
        )
        agent = ClaudeCodeCodingAgent(config)
        agent.initialize(ideation_dir)
        result = agent.generate_code(prompt)
        cost = agent.get_cumulative_cost()
        agent.cleanup()

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
        match = re.search(
            r"<solution>(.*?)</solution>", result.output or "", re.DOTALL
        )
        if result.success and match:
            return {"solution": match.group(1).strip(), "cost_usd": cost}

        # Fail-soft: the pooled work must not die with the selector.
        logger.warning(
            "[GenericSearch] Selector failed "
            f"({result.error or 'no <solution> tag'}); falling back to the "
            "first claude candidate"
        )
        for candidate in pool:
            if candidate["cli"] == "claude_code":
                return {"solution": candidate["text"], "cost_usd": cost}
        return {"solution": pool[0]["text"], "cost_usd": cost}

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
            ideation_repo_memory_sections_consulted: RepoMemory sections used during ideation

        Returns:
            Tuple of (agent output string, phase telemetry with cost/duration)
        """
        from kapso.execution.coding_agents.base import CodingAgentConfig
        from kapso.execution.coding_agents.adapters.claude_code_agent import ClaudeCodeCodingAgent
        from kapso.gated_mcp import get_mcp_config
        from kapso.execution.memories.repo_memory.observation import extract_repo_memory_sections_consulted
        
        # Create experiment session (handles git branching)
        session = self.workspace.create_experiment_session(branch_name, parent_branch_name, llm=self.llm)

        # A maintainer-registered evaluation is versioned on the workspace
        # root, but sessions inherit their parent branch's tree — which may
        # predate a re-registration. Frame-sync the registered tree in so
        # every candidate runs (and is integrity-checked against) the head.
        if self.registered_evaluation_manifest:
            self._sync_registered_evaluation(session.session_folder)
        
        # 1. Load RepoMemory
        repo_memory_doc = RepoMemoryManager.ensure_exists_in_worktree(session.session_folder)
        repo_memory_brief = RepoMemoryManager.render_summary_and_toc(repo_memory_doc, max_chars=2500)
        
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
            "Read", "Write", "Edit", "Bash",
            *[t for t in mcp_tools if t.startswith("mcp__")],
        ]
        
        logger.info(f"[GenericSearch] Implementation tools: {implementation_allowed_tools}")
        
        # 4. Configure Claude Code for implementation
        config = CodingAgentConfig(
            agent_type="claude_code",
            model=self.implementation_model,
            debug_model=self.implementation_model,
            agent_specific={
                **self._claude_auth_settings,
                "env_strip": self.env_strip,
                "aws_region": self.aws_region,
                "mcp_servers": mcp_servers,
                "allowed_tools": implementation_allowed_tools,
                "timeout": self._clamped_timeout(self.implementation_timeout),
                "streaming": True,
                "effort": self.session_effort,
            }
        )
        
        # 5. Build implementation prompt
        repo_memory_detail_access_instructions = (
            "For detailed section content (architecture, gotchas, invariants, etc.),\n"
            "use the MCP tool: `get_repo_memory_section(section_id=\"core.architecture\")`\n"
            "Available sections: core.architecture, core.entrypoints, core.where_to_edit, core.invariants, core.testing, core.gotchas, core.dependencies\n"
            "Fallback: open `.kapso/repo_memory.json` and read `book.sections[section_id]`."
        )
        
        prompt = self._build_implementation_prompt(
            solution=solution,
            problem=problem,
            branch_name=branch_name,
            repo_memory_brief=repo_memory_brief,
            repo_memory_detail_access_instructions=repo_memory_detail_access_instructions,
            previous_errors="\n".join(str(e) for e in self.previous_errors[-self.recent_error_count:]),
        )
        
        # 6. Run Claude Code for implementation
        print(f"[GenericSearch] Running Claude Code implementation...")
        agent = ClaudeCodeCodingAgent(config)
        agent.initialize(session.session_folder)

        phase_started = time.monotonic()
        phase_cost = 0.0
        try:
            result = agent.generate_code(prompt)
            phase_cost = agent.get_cumulative_cost()
            agent_output = result.output if result.output else ""

            # Ground truth about HOW the session ended, for the feedback
            # judge (run #8: a self-inflicted SIGTERM was misdiagnosed as
            # the time limit, so the footgun was never named).
            meta = result.metadata or {}
            if result.success:
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
                agent_output = f"Implementation failed: {result.error}\n\n{agent_output}"
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
            "ideation_repo_memory_sections_consulted": ideation_repo_memory_sections_consulted or [],
        }
        
        # Extract sections consulted from changes.log
        sections_consulted = []
        try:
            changes_log_path = os.path.join(session.session_folder, "changes.log")
            if os.path.exists(changes_log_path):
                with open(changes_log_path, "r", encoding="utf-8", errors="replace") as f:
                    sections_consulted = extract_repo_memory_sections_consulted(f.read())
        except Exception:
            sections_consulted = []
        run_result_payload["repo_memory_sections_consulted"] = sections_consulted
        
        # Schedule RepoMemory update for session close
        session.schedule_repo_memory_update(
            solution_spec=solution,
            run_result=run_result_payload,
        )
        
        # 8. Finalize session (commits changes)
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
        template = load_prompt("execution/search_strategies/generic/prompts/implementation_claude_code.md")
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
        granted_fidelity = (
            decision.eval_fidelity if decision is not None else "full"
        )
        granted_fraction = (
            decision.eval_fraction if decision is not None else 1.0
        )
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

        Only trustworthy measurements become attempts: a registered
        evaluator must exist and the node must carry a valid score.
        """
        if (
            not self.registered_evaluator_id
            or node.score is None
            or node.had_error
            or not node.evaluation_valid
        ):
            return
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
                score=node.score,
                duration_seconds=node.phase_telemetry.get(
                    "implementation", {}
                ).get("duration_seconds"),
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
                commit_sha=self.workspace.repo.commit(
                    target.branch_name
                ).hexsha,
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
            self.record_eval_duration(
                fraction=fraction, duration_seconds=duration
            )
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
2. **Never alter evaluation behavior — at rest or at runtime.** Editing
   anything under `kapso_evaluation/`, rewriting protected data inputs,
   monkey-patching or hooking evaluation modules from your own code
   (e.g. via imports, `sys.modules`, or wrappers), or otherwise
   circumventing any evaluation check all count as tampering: the score
   is voided and the experiment loses. There is no sanctioned bypass.
3. **If you believe the evaluation itself is broken**, do not fix it,
   patch it, or route around it. File a request by including this tag in
   your final response:
   <evaluation_change_request>concrete description of the defect, with the
   exact error output as evidence</evaluation_change_request>
   Then still report your results from the run you attempted. The
   maintainer investigates immediately; a confirmed defect is fixed and
   your work is re-measured first under the corrected evaluation.
4. **Retry on transient crashes** of your own code (max 3 attempts)."""

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
                base_branch=node.parent_branch_name,
                head_branch=node.branch_name,
                evaluation_script_path=node.evaluation_script_path,
                evaluation_result=node.evaluation_output,
                workspace_dir=node.workspace_dir,
                session_end_facts=getattr(
                    self, "_pending_session_end_facts", ""
                ),
                timeout_seconds=self._clamped_timeout(
                    self.feedback_generator.configured_timeout_seconds
                ),
            )

            # Update node with feedback results
            node.feedback = feedback_result.feedback
            node.evaluation_valid = feedback_result.evaluation_valid
            node.score = (
                feedback_result.score
                if feedback_result.evaluation_valid
                else None
            )
            # In registered mode the manifest line is the score of record;
            # the judge's extraction is a cross-check, and the judge keeps
            # its validity power (an invalid evaluation stays scoreless).
            manifest_score = self._manifest_score_of_record(node)
            if manifest_score is not None and node.evaluation_valid:
                if (
                    node.score is not None
                    and abs(node.score - manifest_score) > 1e-6
                ):
                    print(
                        "[GenericSearch] Score cross-check: feedback "
                        f"extracted {node.score}, the manifest says "
                        f"{manifest_score}; the manifest is the score "
                        "of record"
                    )
                node.score = manifest_score
            node.should_stop = (
                feedback_result.stop and feedback_result.evaluation_valid
            )
            if feedback_result.duration_seconds is not None:
                node.phase_telemetry["feedback"] = {
                    "cost_usd": feedback_result.cost_usd,
                    "duration_seconds": feedback_result.duration_seconds,
                }
            
            print(f"[GenericSearch] Feedback generated: stop={node.should_stop}, score={node.score}")
            
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
        
        Args:
            agent_output: Raw output from the developer agent
            
        Returns:
            dict with keys: code_changes_summary, evaluation_script_path, evaluation_output, score
            Returns empty dict if extraction fails
        """
        result = {}
        
        # Extract each tag
        tags = ["code_changes_summary", "evaluation_script_path", "evaluation_output", "score"]
        
        for tag in tags:
            pattern = rf'<{tag}>\s*(.*?)\s*</{tag}>'
            match = re.search(pattern, agent_output, re.DOTALL)
            if match:
                value = match.group(1).strip()
                # Handle score specially - convert to float
                if tag == "score":
                    try:
                        if value.lower() == "null" or value == "":
                            result[tag] = None
                        else:
                            result[tag] = float(value)
                    except ValueError:
                        result[tag] = None
                else:
                    result[tag] = value
        
        if result:
            print(f"[GenericSearch] Extracted agent result from XML tags: {list(result.keys())}")
            return result
        
        # Fallback: try JSON extraction for backward compatibility
        return self._extract_agent_result_json_fallback(agent_output)
    
    def _extract_agent_result_json_fallback(self, agent_output: str) -> dict:
        """
        Fallback JSON extraction for backward compatibility.
        """
        # Look for JSON in code blocks (```json ... ```)
        json_pattern = r'```json\s*(\{.*?\})\s*```'
        matches = re.findall(json_pattern, agent_output, re.DOTALL)
        
        if matches:
            # Take the last JSON block (final result)
            for json_str in reversed(matches):
                try:
                    result = json.loads(json_str)
                    # Validate it has expected keys
                    if any(k in result for k in ["code_changes_summary", "evaluation_output", "evaluation_script_path"]):
                        print(f"[GenericSearch] Extracted agent result from JSON block (fallback)")
                        return result
                except json.JSONDecodeError:
                    continue
        
        # Fallback: try to find raw JSON object at the end
        try:
            # Find last occurrence of {...}
            start = agent_output.rfind('{')
            end = agent_output.rfind('}') + 1
            if start != -1 and end > start:
                json_str = agent_output[start:end]
                result = json.loads(json_str)
                if any(k in result for k in ["code_changes_summary", "evaluation_output", "evaluation_script_path"]):
                    print(f"[GenericSearch] Extracted agent result from raw JSON (fallback)")
                    return result
        except json.JSONDecodeError:
            pass
        
        print(f"[GenericSearch] Warning: Could not extract result from agent output")
        return {}

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
        if iteration_count < len(self.node_history):
            raise ValueError(
                "GenericSearch checkpoint iteration_count cannot be smaller "
                "than node_history: every node consumed an iteration"
            )
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
