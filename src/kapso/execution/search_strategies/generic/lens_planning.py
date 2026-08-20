"""Task-aware ideation lens planning for the generic strategy.

Owns the lens planner/replanner feature: config normalization and
validation, plan/revision parsing, the planner claude session, per-campaign
plan + history file IO, and the design-axes / member-roster prompt briefs.
Stateless functions only — GenericSearch assembles arguments from its state
and delegates here.
"""

import json
import logging
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from kapso.core.prompt_loader import load_prompt, render_prompt
from kapso.execution.search_strategies.base import SearchNode
from kapso.execution.search_strategies.generic import codex_ideation

logger = logging.getLogger(__name__)

_LENS_PLANNER_KEYS = frozenset({"cli", "model", "effort", "timeout"})
LENS_PLAN_FILENAME = "lens_plan.json"
LENS_PLAN_HISTORY_FILENAME = "lens_plan_history.jsonl"


DESIGN_AXES_DEFAULT: Tuple[str, ...] = (
    "input representation — the features/joins/encodings fed to the model",
    "training distribution — example construction, augmentation, weighting",
    "model mechanism — estimator family, objectives, ensembling",
    "decoding / post-processing — calibration, constraints, blending",
    "validation protocol — splits/origins, gates, generalization estimation",
)


def normalize_design_axes(value: Any) -> Tuple[str, ...]:
    """Task-declared design axes of the solution space.

    The axes feed the lens planner/replanner axis-coverage contract and the
    feedback generator's axis-frontier report. None selects the generic
    default set; a task supplies its own vocabulary via the mode config.
    """
    if value is None:
        return DESIGN_AXES_DEFAULT
    if not isinstance(value, list) or not value:
        raise ValueError("design_axes must be a non-empty list of strings")
    axes: List[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("design_axes entries must be non-empty strings")
        axes.append(item.strip())
    return tuple(axes)


def normalize_ideation_lens_planner(value: Any) -> Optional[Dict[str, Any]]:
    """Validate the optional task-aware lens planner config block."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("ideation_lens_planner must be a mapping")
    unknown = sorted(set(value) - _LENS_PLANNER_KEYS)
    if unknown:
        raise ValueError(
            f"ideation_lens_planner has unknown keys: {', '.join(unknown)}"
        )
    if value.get("cli") != "claude_code":
        raise ValueError(
            "ideation_lens_planner.cli must be claude_code (the planner "
            "needs the CLI's native WebSearch/WebFetch tools)"
        )
    model = value.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("ideation_lens_planner.model must be a non-empty string")
    timeout = value.get("timeout")
    if timeout is not None and (isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0):
        raise ValueError("ideation_lens_planner.timeout must be a positive number")
    return dict(value)


def validate_lens_planner_against_ensemble(
    planner: Optional[Dict[str, Any]],
    ensemble: Optional[List[Dict[str, str]]],
) -> None:
    """The planner owns every lens: static member lenses are forbidden with it."""
    if planner is None:
        return
    if not ensemble:
        raise ValueError(
            "ideation_lens_planner requires ideation_ensemble (there are no "
            "members to design lenses for)"
        )
    lensed = [i for i, member in enumerate(ensemble) if "lens" in member]
    if lensed:
        raise ValueError(
            "ideation_ensemble members "
            f"{', '.join(str(i) for i in lensed)} carry static lens keys — "
            "remove them (the lens planner designs every lens) or disable "
            "ideation_lens_planner"
        )


def parse_lens_plan(output: str, expected_count: int) -> Dict[str, Any]:
    """Extract <lens_N> tags (member order) + <sources> from planner output."""
    lenses = []
    for i in range(1, expected_count + 1):
        match = re.search(
            rf"<lens_{i}>(.*?)</lens_{i}>", output, re.DOTALL
        )
        if not match or not match.group(1).strip():
            raise ValueError(
                f"lens planner output missing a non-empty <lens_{i}> tag "
                f"(expected {expected_count} lenses)"
            )
        lenses.append(" ".join(match.group(1).split()))
    sources_match = re.search(r"<sources>(.*?)</sources>", output, re.DOTALL)
    return {
        "lenses": lenses,
        "sources": sources_match.group(1).strip() if sources_match else "",
    }


def parse_lens_revision(output: str, expected_count: int) -> Dict[str, Any]:
    """Classify a keep-or-revise replanner output; never raises.

    Returns one of:
      {"kind": "keep", "rationale": str}
      {"kind": "revise", "lenses": [...], "sources": str, "rationale": str}
      {"kind": "invalid", "reason": str}
    `invalid` is a first-class outcome: the caller records it loudly and the
    previous validated plan stays in force — a mid-flight campaign must
    never die on a malformed revision.
    """
    text = output or ""
    keep = re.search(r"<keep>(.*?)</keep>", text, re.DOTALL)
    if keep and keep.group(1).strip():
        return {"kind": "keep", "rationale": " ".join(keep.group(1).split())}
    lenses = []
    for i in range(1, expected_count + 1):
        match = re.search(rf"<lens_{i}>(.*?)</lens_{i}>", text, re.DOTALL)
        if not match or not match.group(1).strip():
            return {
                "kind": "invalid",
                "reason": (
                    "neither a non-empty <keep> nor a complete lens set "
                    f"(missing <lens_{i}> of {expected_count})"
                ),
            }
        lenses.append(" ".join(match.group(1).split()))
    rationale = re.search(
        r"<revision_rationale>(.*?)</revision_rationale>", text, re.DOTALL
    )
    sources = re.search(r"<sources>(.*?)</sources>", text, re.DOTALL)
    return {
        "kind": "revise",
        "lenses": lenses,
        "sources": sources.group(1).strip() if sources else "",
        "rationale": (
            " ".join(rationale.group(1).split())
            if rationale and rationale.group(1).strip()
            else ""
        ),
    }


def design_axes_brief(design_axes: Tuple[str, ...]) -> str:
    """Numbered design-axes block for prompts."""
    return "\n".join(
        f"{i}. {axis}" for i, axis in enumerate(design_axes, 1)
    )


def member_roster_brief(ideation_ensemble: List[Dict[str, str]]) -> str:
    """One line per ensemble member for the planner prompts."""
    return "\n".join(
        f"- member {i + 1}: cli={m['cli']}, model={m['model']}"
        + (
            " (has native web search during ideation)"
            if m["cli"] == "codex"
            else ""
        )
        for i, m in enumerate(ideation_ensemble)
    )


def run_lens_planner_session(
    prompt: str,
    ideation_dir: str,
    *,
    planner: Dict[str, Any],
    claude_auth_settings: Dict[str, Any],
    env_strip: List[str],
    env_defaults: Dict[str, str],
    aws_region: str,
    web_disallowed_tools: List[str],
    ideation_web_search: bool,
    session_effort: Optional[str],
    artifacts_dir: str,
):
    """One planner/replanner claude session; returns (result, cost_usd)."""
    from kapso.execution.coding_agents.base import CodingAgentConfig
    from kapso.execution.coding_agents.adapters.claude_code_agent import ClaudeCodeCodingAgent

    print(
        f"[GenericSearch] Lens planner starting: {planner['model']} "
        f"({'web-enabled' if ideation_web_search else 'web-OFF'})"
    )
    config = CodingAgentConfig(
        agent_type="claude_code",
        model=planner["model"],
        debug_model=planner["model"],
        agent_specific={
            **claude_auth_settings,
            "env_strip": env_strip,
            "env_defaults": env_defaults,
            "aws_region": aws_region,
            "allowed_tools": ["Read", "WebSearch", "WebFetch"],
            "disallowed_tools": web_disallowed_tools,
            "timeout": planner.get("timeout", 600),
            "streaming": True,
            "planning_mode": False,
            "effort": planner.get("effort", session_effort),
            "stream_artifact_path": codex_ideation.ideation_stream_path(
                artifacts_dir, "lens_planner",
                planner["model"],
            ),
        },
    )
    agent = ClaudeCodeCodingAgent(config)
    agent.initialize(ideation_dir)
    result = agent.generate_code(prompt)
    cost = agent.get_cumulative_cost()
    agent.cleanup()
    return result, cost


def append_lens_history(history_path: str, record: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def write_lens_plan(plan_path: str, plan: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(plan_path), exist_ok=True)
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)


def resolve_member_lenses(
    problem: str,
    ideation_dir: str,
    *,
    ideation_lens_planner: Optional[Dict[str, Any]],
    ideation_ensemble: List[Dict[str, str]],
    workspace_dir: str,
    iteration_count: int,
    shared_artifacts_brief: str,
    design_axes: Tuple[str, ...],
    node_history: List[SearchNode],
    campaign_state_brief: Callable[[], str],
    get_best_experiment: Callable[[], Optional[SearchNode]],
    run_planner_session: Callable[[str, str], tuple],
) -> tuple:
    """Task-aware lenses, replanned per iteration; returns (lenses, cost).

    (None, 0.0) when no planner block is configured (static config
    lenses). Iteration 1 runs the full research planner — fail-loud, a
    restart is cheap. Every LATER iteration runs a keep-or-revise session
    over the campaign evidence (state brief, recent judge feedback, the
    champion's solution): the plan is re-aimed when the evidence says the
    current angles are exhausted, kept when they still carry the highest
    credible return toward the bar. A failed or unparseable revision
    falls back LOUDLY to the previous validated plan — it never kills a
    mid-flight campaign. The current plan lives in .kapso/lens_plan.json
    (keyed by the iteration that last confirmed it, so a same-iteration
    resume reuses it without a session); every planner decision appends
    to .kapso/lens_plan_history.jsonl as the audit trail.
    """
    if not ideation_lens_planner:
        return None, 0.0
    expected = len(ideation_ensemble)
    kapso_dir = os.path.join(workspace_dir, ".kapso")
    plan_path = os.path.join(kapso_dir, LENS_PLAN_FILENAME)
    history_path = os.path.join(kapso_dir, LENS_PLAN_HISTORY_FILENAME)

    plan = None
    if os.path.isfile(plan_path):
        with open(plan_path, encoding="utf-8") as f:
            plan = json.load(f)
        if len(plan["lenses"]) != expected:
            raise ValueError(
                f"{plan_path} holds {len(plan['lenses'])} lenses for "
                f"{expected} ensemble members — delete it to replan"
            )
        if plan.get("iteration") == iteration_count:
            return plan["lenses"], 0.0

    planner = ideation_lens_planner
    roster = member_roster_brief(ideation_ensemble)

    if plan is None:
        prompt = render_prompt(
            load_prompt(
                "execution/search_strategies/generic/prompts/ideation_lens_planner.md"
            ),
            {
                "problem": problem,
                "member_roster": roster,
                "lens_count": str(expected),
                "shared_artifacts_brief": shared_artifacts_brief,
                "design_axes": design_axes_brief(design_axes),
            },
        )
        result, cost = run_planner_session(prompt, ideation_dir)
        if not result.success:
            raise RuntimeError(
                f"lens planner session failed: {result.error}"
            )
        parsed = parse_lens_plan(result.output, expected)
        plan = {
            "lenses": parsed["lenses"],
            "sources": parsed["sources"],
            "planner_model": planner["model"],
            "iteration": iteration_count,
            "decision": "initial",
            "rationale": "",
        }
        write_lens_plan(plan_path, plan)
        append_lens_history(history_path, plan)
        for i, lens in enumerate(plan["lenses"], 1):
            print(f"[GenericSearch] Lens {i}: {lens}")
        return plan["lenses"], cost

    # Keep-or-revise: the replanner judges the plan against the campaign
    # evidence. Judge feedbacks and the champion solution go in FULL —
    # content bound for a model call is never truncated.
    feedbacks = [
        node.feedback
        for node in node_history
        if getattr(node, "feedback", None)
    ]
    recent_feedback = (
        "\n\n---\n\n".join(feedbacks[-2:])
        if feedbacks
        else "(no judge feedback yet)"
    )
    best = get_best_experiment()
    champion_solution = (
        best.solution
        if best is not None and best.solution
        else "(no scored champion yet)"
    )
    previous_lenses = "\n".join(
        f"lens {i}: {lens}" for i, lens in enumerate(plan["lenses"], 1)
    )
    prompt = render_prompt(
        load_prompt(
            "execution/search_strategies/generic/prompts/ideation_lens_replanner.md"
        ),
        {
            "problem": problem,
            "member_roster": roster,
            "lens_count": str(expected),
            "shared_artifacts_brief": shared_artifacts_brief,
            "design_axes": design_axes_brief(design_axes),
            "campaign_state": campaign_state_brief(),
            "plan_iteration": str(plan.get("iteration", "?")),
            "previous_lenses": previous_lenses,
            "previous_sources": plan.get("sources") or "(none recorded)",
            "previous_rationale": plan.get("rationale") or "(none recorded)",
            "recent_feedback": recent_feedback,
            "champion_solution": champion_solution,
        },
    )
    result, cost = run_planner_session(prompt, ideation_dir)
    revision = (
        parse_lens_revision(result.output, expected)
        if result.success
        else {"kind": "invalid", "reason": f"session failed: {result.error}"}
    )
    if revision["kind"] == "revise":
        plan = {
            "lenses": revision["lenses"],
            "sources": revision["sources"],
            "planner_model": planner["model"],
            "iteration": iteration_count,
            "decision": "revise",
            "rationale": revision["rationale"],
        }
        write_lens_plan(plan_path, plan)
        append_lens_history(history_path, plan)
        print(
            f"[GenericSearch] Lens plan REVISED (iteration "
            f"{iteration_count}): {plan['rationale']}"
        )
        for i, lens in enumerate(plan["lenses"], 1):
            print(f"[GenericSearch] Lens {i}: {lens}")
        return plan["lenses"], cost
    if revision["kind"] == "keep":
        plan["iteration"] = iteration_count
        plan["decision"] = "keep"
        plan["rationale"] = revision["rationale"]
        write_lens_plan(plan_path, plan)
        append_lens_history(
            history_path,
            {
                "iteration": iteration_count,
                "decision": "keep",
                "rationale": revision["rationale"],
                "lenses": plan["lenses"],
            },
        )
        print(
            f"[GenericSearch] Lens plan kept (iteration "
            f"{iteration_count}): {revision['rationale']}"
        )
        return plan["lenses"], cost
    # invalid: loud fallback to the previous validated plan. The plan
    # file's iteration is NOT bumped, so a same-iteration retry replans.
    logger.warning(
        "[GenericSearch] Lens revision invalid "
        f"({revision['reason']}); keeping previous plan"
    )
    append_lens_history(
        history_path,
        {
            "iteration": iteration_count,
            "decision": "failed",
            "reason": revision["reason"],
            "raw_output": result.output or "",
        },
    )
    return plan["lenses"], cost
