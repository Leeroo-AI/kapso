"""Ensemble and single-session ideation for the generic strategy.

Owns the ideation feature: member config normalization and degeneracy
hygiene, the single Claude session path, the parallel CLI-member fan-out
with per-member transcripts, the selector-critic session (including the
malformed-emission retry and pool top-up hardening), the selector's
ranked-solution parser, the campaign-state brief, ideation prompt build,
and output extraction / salvage / fallback. Stateless functions only —
GenericSearch assembles arguments from its state and delegates here.
"""

import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple

from kapso.core.prompt_loader import load_prompt, render_prompt
from kapso.execution.memories.repo_memory import RepoMemoryManager
from kapso.execution.search_strategies.base import SearchNode
from kapso.execution.search_strategies.generic import codex_ideation

logger = logging.getLogger(__name__)

# A deadline-killed ideation whose streamed text is shorter than this holds
# no consumable plan; the explicit fallback is more honest than salvage.
MIN_IDEATION_SALVAGE_CHARS = 200

# Default only — the live value is the optional selector_min_seconds key
# of search_strategy.params ensemble_time_split (normalize_ensemble_time_split).
# The floor exists because below it a read-verify-choose selector session
# cannot do useful work; it applies ONLY while a split is active — the
# no-split default gives each role the full ideation clamp, floor-free.
ENSEMBLE_SPLIT_SELECTOR_MIN_SECONDS = 240
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


_ENSEMBLE_TIME_SPLIT_KEYS = frozenset(
    {"member_fraction", "selector_fraction", "selector_min_seconds"}
)


def normalize_ensemble_time_split(value: Any) -> Optional[Dict[str, float]]:
    """Validate the optional member/selector ideation time split (design #5).

    None — the platform default — means NO split: members and the selector
    each take the full clamped ideation timeout, and the selector's clamp
    is recomputed after the members finish so time they did not spend
    flows to it. A mapping like {member_fraction: 0.7,
    selector_fraction: 0.3} restores the fractional split of the
    pre-fan-out clamp; its selector floor (selector_min_seconds, default
    ENSEMBLE_SPLIT_SELECTOR_MIN_SECONDS) applies only while the split is
    active.
    """
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(
            "ensemble_time_split must be a mapping, got "
            f"{type(value).__name__}"
        )
    unknown = sorted(set(value) - _ENSEMBLE_TIME_SPLIT_KEYS)
    if unknown:
        raise ValueError(
            f"ensemble_time_split has unknown keys: {unknown}"
        )
    normalized = {}
    for key in ("member_fraction", "selector_fraction"):
        if key not in value:
            raise ValueError(f"ensemble_time_split requires {key}")
        fraction = value[key]
        if (
            isinstance(fraction, bool)
            or not isinstance(fraction, (int, float))
            or not 0 < fraction <= 1
        ):
            raise ValueError(
                f"ensemble_time_split.{key} must be a fraction in (0, 1], "
                f"got {fraction!r}"
            )
        normalized[key] = float(fraction)
    floor = value.get(
        "selector_min_seconds", ENSEMBLE_SPLIT_SELECTOR_MIN_SECONDS
    )
    if (
        isinstance(floor, bool)
        or not isinstance(floor, (int, float))
        or floor < 0
    ):
        raise ValueError(
            "ensemble_time_split.selector_min_seconds must be a "
            f"non-negative number, got {floor!r}"
        )
    normalized["selector_min_seconds"] = float(floor)
    return normalized


def generate_solution(
    problem: str,
    parent_branch: str,
    *,
    workspace,
    llm,
    experiment_history_path: str,
    ideation_gates: List[str],
    gate_failure_policy: str,
    bank_serving: Optional[Dict[str, str]] = None,
    kg_index_path: Optional[str] = None,
    ideation_web_search: bool,
    ideation_ensemble: Optional[List[Dict[str, str]]],
    idea_generation_model: str,
    claude_auth_settings: Dict[str, Any],
    env_strip: List[str],
    env_defaults: Dict[str, str],
    aws_region: str,
    web_disallowed_tools: List[str],
    clamped_timeout: Callable[[float], float],
    ideation_timeout: float,
    session_effort: Optional[str],
    build_prompt: Callable[..., str],
    run_ensemble: Callable[..., Tuple[List[str], List[str], Dict[str, float]]],
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
        workspace.repo, parent_branch
    ) or {}
    repo_memory_brief = RepoMemoryManager.render_summary_and_toc(
        repo_memory_doc, max_chars=2500
    )
    
    # Materialize the selected ref without changing the root workspace's
    # checkout. Every read-only ideation surface points at this same tree.
    with workspace.materialize_ref(parent_branch) as ideation_dir:
        # 2. Configure gates against the selected parent tree. Keep the
        # history path absolute because the MCP process may run elsewhere.
        mcp_servers, mcp_tools = get_mcp_config(
            gates=ideation_gates,
            experiment_history_path=os.path.abspath(
                experiment_history_path
            ),
            experiment_embedding_model=(
                llm.resolve_model(None, default_role="embedding")
                if llm is not None
                else None
            ),
            repo_root=ideation_dir,
            include_base_tools=False,
            gate_failure_policy=gate_failure_policy,
            bank_serving=bank_serving,
            kg_index_path=kg_index_path,
            research_web_search_model=(
                llm.resolve_model(None, default_role="web_search")
                if llm is not None else None
            ),
        )

        # 3. Build restricted tool set (read-only for ideation). Claude
        # CLIs research with their NATIVE web tools — so WebSearch/
        # WebFetch join the whitelist whenever ideation web access is on
        # (the research_* gate proxies coexist; gates decide availability).
        ideation_allowed_tools = [
            "Read",
            *(["WebSearch", "WebFetch"] if ideation_web_search else []),
            *[t for t in mcp_tools if t.startswith("mcp__")],
        ]

        logger.info(
            f"[GenericSearch] Ideation tools: {ideation_allowed_tools}"
        )

        if ideation_ensemble:
            return run_ensemble(
                problem=problem,
                repo_memory_brief=repo_memory_brief,
                ideation_dir=ideation_dir,
                mcp_servers=mcp_servers,
                ideation_allowed_tools=ideation_allowed_tools,
            )

        # 4. Configure Claude Code for ideation (read-only mode).
        config = CodingAgentConfig(
            agent_type="claude_code",
            model=idea_generation_model,
            debug_model=idea_generation_model,
            agent_specific={
                **claude_auth_settings,
                "env_strip": env_strip,
                "env_defaults": env_defaults,
                "aws_region": aws_region,
                "mcp_servers": mcp_servers,
                "allowed_tools": ideation_allowed_tools,
                "disallowed_tools": web_disallowed_tools,
                "timeout": clamped_timeout(ideation_timeout),
                "streaming": True,
                "planning_mode": False,
                "effort": session_effort,
            },
        )

        # 5. Build the ideation prompt.
        prompt = build_prompt(
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
                salvaged = salvage_ideation_output(result)
                if salvaged is not None:
                    print(
                        "[GenericSearch] Salvaged partial output "
                        f"({len(salvaged)} chars) from the "
                        "deadline-terminated ideation session"
                    )
                    return (
                        [salvaged],
                        extract_sections_consulted(result.output),
                        telemetry,
                    )
                return [fallback_solution(problem)], [], telemetry

            solution = extract_solution_from_output(result.output)
            sections_consulted = extract_sections_consulted(
                result.output
            )

            print(
                "[GenericSearch] Ideation complete, sections consulted: "
                f"{sections_consulted}"
            )
            return [solution], sections_consulted, telemetry
        finally:
            agent.cleanup()


def generate_solution_ensemble(
    *,
    problem: str,
    repo_memory_brief: str,
    ideation_dir: str,
    mcp_servers: Dict[str, Any],
    ideation_allowed_tools: List[str],
    ideation_ensemble: List[Dict[str, str]],
    ideation_candidates_per_member: int,
    ensemble_time_split: Optional[Dict[str, float]],
    ideation_web_search: bool,
    claude_auth_settings: Dict[str, Any],
    env_strip: List[str],
    env_defaults: Dict[str, str],
    aws_region: str,
    web_disallowed_tools: List[str],
    session_effort: Optional[str],
    clamped_timeout: Callable[[float], float],
    ideation_timeout: float,
    artifacts_dir: str,
    build_prompt: Callable[..., str],
    resolve_lenses: Callable[[str, str], tuple],
    select_candidates: Callable[..., Dict[str, Any]],
) -> Tuple[str, List[str], Dict[str, float]]:
    """Fan out ideation across CLI members, then select one solution.

    Members run in parallel (they are API-bound, never GPU-bound) inside
    the same read-only worktree; a selector-critic session chooses among
    the pooled <solution> candidates. Fail-soft ladder: selector failure
    -> first claude_code candidate -> any candidate -> template fallback.
    """
    phase_started = time.monotonic()

    base_prompt = build_prompt(
        problem=problem, repo_memory_brief=repo_memory_brief
    )
    addendum_template = load_prompt(
        "execution/search_strategies/generic/prompts/ideation_ensemble_addendum.md"
    )

    member_lenses, lens_planner_cost = resolve_lenses(
        problem, ideation_dir
    )

    # Deadlines are computed AFTER the planner session so its wall time
    # squeezes this iteration's members instead of overflowing the phase.
    # Default (no ensemble_time_split): each role takes the FULL ideation
    # clamp — the campaign budget, not a fraction, is the limit; the
    # selector's clamp is recomputed after the members finish so time they
    # did not spend flows to it instead of being forfeited. An explicit
    # split restores the fractional carve-up of this pre-fan-out clamp.
    clamp = clamped_timeout(ideation_timeout)
    member_deadline = (
        clamp
        if ensemble_time_split is None
        else max(60.0, clamp * ensemble_time_split["member_fraction"])
    )

    def run_member(member: Dict[str, str], lens: str) -> Dict[str, Any]:
        prompt = base_prompt + "\n\n" + render_prompt(
            addendum_template,
            {
                "lens": lens,
                "candidate_count": str(ideation_candidates_per_member),
            },
        )
        label = f"{member['cli']}:{member['model']}"
        print(f"[GenericSearch] Ensemble ideation member starting: {label}")
        # Every member persists its transcript here, not just codex: the
        # claude-driven members used to stream to the console only, so
        # their reasoning survived just in whatever wrapper happened to
        # capture stdout.
        if member["cli"] == "codex":

            def run_codex_once(attempt_deadline: float) -> tuple:
                return codex_ideation.run_codex_ideation(
                    prompt=prompt,
                    model=member["model"],
                    cwd=ideation_dir,
                    timeout_seconds=attempt_deadline,
                    effort=member.get("effort"),
                    artifacts_dir=artifacts_dir,
                    web_search=ideation_web_search,
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
                    + extract_solution_from_output(output.strip())
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
            "env_strip": env_strip,
            "env_defaults": env_defaults,
            "aws_region": aws_region,
            "mcp_servers": mcp_servers,
            "allowed_tools": member_allowed_tools,
            "disallowed_tools": web_disallowed_tools,
            "timeout": member_deadline,
            "streaming": True,
            "planning_mode": False,
            "effort": member.get("effort", session_effort),
            "stream_artifact_path": codex_ideation.ideation_stream_path(
                artifacts_dir, member["cli"], member["model"]
            ),
        }
        if is_oss:
            # Endpoint wiring replaces first-party auth entirely.
            agent_specific["base_url"] = member["base_url"]
            agent_specific["auth_token_env"] = member["auth_token_env"]
        else:
            agent_specific.update(claude_auth_settings)
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
            salvaged = salvage_ideation_output(result)
            candidates = [salvaged] if salvaged is not None else []
        else:
            candidates = [
                c.strip()
                for c in re.findall(
                    r"<solution>(.*?)</solution>", result.output, re.DOTALL
                )
            ] or [extract_solution_from_output(result.output)]
        return {
            "label": label,
            "cli": "claude_code",
            "candidates": candidates,
            "sections": extract_sections_consulted(result.output),
            "cost_usd": cost,
        }

    members = ideation_ensemble
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
            f"candidates={kept}/{ideation_candidates_per_member} "
            f"(dropped {dropped}){timing}, "
            f"timed_out={member_result.get('timed_out', False)}, {detail}"
        )
        if kept < ideation_candidates_per_member:
            logger.warning(
                f"[GenericSearch] member {member_result['label']} "
                f"under-delivered: {kept} of "
                f"{ideation_candidates_per_member} candidates"
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
        return [fallback_solution(problem)], sections, telemetry
    if len(pool) == 1:
        print("[GenericSearch] Single candidate — selector skipped")
        return [pool[0]["text"]], sections, telemetry

    selector_deadline = (
        clamped_timeout(ideation_timeout)
        if ensemble_time_split is None
        else max(
            ensemble_time_split["selector_min_seconds"],
            clamp * ensemble_time_split["selector_fraction"],
        )
    )
    chosen = select_candidates(
        problem=problem,
        repo_memory_brief=repo_memory_brief,
        pool=pool,
        ideation_dir=ideation_dir,
        selector_deadline=selector_deadline,
    )
    telemetry["cost_usd"] += chosen["cost_usd"]
    telemetry["duration_seconds"] = time.monotonic() - phase_started
    return chosen["solutions"], sections, telemetry


def campaign_state_brief(
    node_history: List[SearchNode], maximize_scoring: bool
) -> str:
    """Factual campaign trajectory for the selector's return judgment.

    The selector prompt values candidates by expected return against the
    GOAL's bar; this supplies the other half of that arithmetic — where
    the campaign currently stands and whether progress has stalled.
    """
    scored = [
        n
        for n in node_history
        if not n.had_error and n.evaluation_valid and n.score is not None
    ]
    if not scored:
        return (
            "No scored experiments yet — the pool below is the campaign's "
            "first swing; judge return against the GOAL's published bar."
        )
    maximize = maximize_scoring
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


def select_from_candidates(
    *,
    problem: str,
    repo_memory_brief: str,
    pool: List[Dict[str, str]],
    ideation_dir: str,
    selector_deadline: float,
    ideation_selector: Dict[str, str],
    node_expansion_value: int,
    campaign_state: str,
    claude_auth_settings: Dict[str, Any],
    env_strip: List[str],
    env_defaults: Dict[str, str],
    aws_region: str,
    session_effort: Optional[str],
    artifacts_dir: str,
) -> Dict[str, Any]:
    """Run the selector-critic session over the pooled candidates.

    Returns {"solutions": List[str] (rank order, len<=node_expansion_value),
    "cost_usd": float}. With expansion 1 this is today's single pick.
    """
    from kapso.execution.coding_agents.base import CodingAgentConfig
    from kapso.execution.coding_agents.adapters.claude_code_agent import ClaudeCodeCodingAgent

    expansion = node_expansion_value
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
            "campaign_state": campaign_state,
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
    selector = ideation_selector

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
                artifacts_dir=artifacts_dir,
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
                **claude_auth_settings,
                "env_strip": env_strip,
                "env_defaults": env_defaults,
                "aws_region": aws_region,
                "allowed_tools": ["Read", "WebSearch", "WebFetch"],
                "timeout": selector_deadline,
                "streaming": True,
                "planning_mode": False,
                "effort": selector.get("effort", session_effort),
                "stream_artifact_path": codex_ideation.ideation_stream_path(
                    artifacts_dir, "selector",
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


def build_ideation_prompt(
    problem: str,
    repo_memory_brief: str,
    *,
    budget_status: str,
    shared_artifacts_brief: str,
) -> str:
    """Build the ideation prompt for Claude Code."""
    # Load and render the prompt template
    template = load_prompt("execution/search_strategies/generic/prompts/ideation_claude_code.md")
    return render_prompt(
        template,
        {
            "problem": problem or "(No problem description provided)",
            "repo_memory_brief": repo_memory_brief or "(No repo memory available)",
            "budget_status": budget_status,
            "shared_artifacts_brief": shared_artifacts_brief,
        },
    )


def extract_solution_from_output(output: str) -> str:
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


def extract_sections_consulted(output: str) -> List[str]:
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


def salvage_ideation_output(result) -> Optional[str]:
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
        f"{extract_solution_from_output(partial)}"
    )


def fallback_solution(problem: str) -> str:
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
