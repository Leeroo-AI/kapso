"""Task-aware ideation lens planning for the generic strategy.

Owns the lens planner/replanner feature: the planner's validation,
plan/revision parsing (the design axes the planner names included), the
planner session, per-campaign plan + history file IO, and the axes /
member-roster prompt briefs.
Stateless functions only — GenericSearch assembles arguments from its state
and delegates here.
"""

import json
import logging
import os
import re
from typing import Any, Callable, Dict, List, Optional

from kapso.core.prompt_loader import load_prompt, optional_block, render_prompt
from kapso.harness.staging import StagedHarness
from kapso.execution.search_strategies.base import SearchNode
from kapso.execution.search_strategies.generic import codex_ideation
from kapso.execution.search_strategies.generic.ideation import WEB_TOOLS
from kapso.core.roles import SessionRole
from kapso.execution.coding_agents.factory import CodingAgentFactory

logger = logging.getLogger(__name__)

LENS_PLAN_FILENAME = "lens_plan.json"
LENS_PLAN_HISTORY_FILENAME = "lens_plan_history.jsonl"


# The design axes of the solution space are the planner's: its first plan
# names them (derived from the goal and the ideation page), a revision
# restates every one with its status, and the judge reports the frontier
# per axis while a plan exists (docs/plans/redesign/wp4-config.md).
AXIS_LINE = re.compile(r"^\s*(\d+)[.)]\s*(\S.*?)\s*$")


def numbered_list(text: str) -> Optional[List[str]]:
    """The items of a list numbered from 1; None when a line is not."""
    items: List[str] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        item = AXIS_LINE.match(line)
        if item is None or int(item.group(1)) != len(items) + 1:
            return None
        items.append(item.group(2))
    return items


def parse_axes(output: str, origin: str) -> List[str]:
    """The `<axes>` tag's list; raises when the tag is missing, empty or
    not numbered from 1 (the first plan fails loud, a restart is cheap)."""
    match = re.search(r"<axes>(.*?)</axes>", output or "", re.DOTALL)
    if match is None:
        raise ValueError(f"{origin}: no <axes> tag")
    axes = numbered_list(match.group(1))
    if axes is None:
        raise ValueError(f"{origin}: <axes> is not a list numbered from 1")
    if not axes:
        raise ValueError(f"{origin}: <axes> names no axis")
    return axes


def validated_axes(value: Any, origin: str) -> List[str]:
    """A plan record's axes: a non-empty list of non-empty strings."""
    if (
        not isinstance(value, list) or not value
        or any(not isinstance(axis, str) or not axis.strip() for axis in value)
    ):
        raise ValueError(f"{origin} has no axes — delete it to replan")
    return list(value)


def axes_brief(axes: List[str]) -> str:
    """Numbered axes block for prompts; empty without axes."""
    return "\n".join(f"{i}. {axis}" for i, axis in enumerate(axes, 1))


def current_plan_axes(workspace_dir: str) -> List[str]:
    """The lens plan's axes, for the judge's frontier duty: empty when the
    campaign has no plan (no planner, or none written yet)."""
    plan_path = os.path.join(workspace_dir, ".kapso", LENS_PLAN_FILENAME)
    if not os.path.isfile(plan_path):
        return []
    with open(plan_path, encoding="utf-8") as f:
        return validated_axes(json.load(f).get("axes"), plan_path)


def validate_lens_planner(
    planner: Optional[SessionRole],
    ensemble: Optional[List[SessionRole]],
) -> None:
    """The planner researches with Claude Code's native web tools and owns
    every lens: a Codex host is refused, and static member lenses are
    forbidden with it."""
    if planner is None:
        return
    if planner.cli == "codex":
        raise ValueError(
            "search_strategy.params.ideation.lens_planner.cli: the planner researches "
            "with Claude Code's native WebSearch/WebFetch tools; name a Claude Code host"
        )
    if not ensemble:
        raise ValueError(
            "search_strategy.params.ideation.lens_planner needs an ensemble (there are "
            "no members to design lenses for)"
        )
    lensed = [i for i, member in enumerate(ensemble) if member.lens is not None]
    if lensed:
        raise ValueError(
            "search_strategy.params.ideation.ensemble members "
            f"{', '.join(str(i) for i in lensed)} carry static lens keys — "
            "remove them (the lens planner designs every lens) or drop the lens_planner"
        )


def parse_lens_plan(output: str, expected_count: int) -> Dict[str, Any]:
    """The <axes>, the <lens_N> tags (member order) and <sources> of the
    planner's output."""
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
        "axes": parse_axes(output, "lens planner output"),
        "lenses": lenses,
        "sources": sources_match.group(1).strip() if sources_match else "",
    }


def parse_lens_revision(output: str, expected_count: int, axes_count: int) -> Dict[str, Any]:
    """Classify a keep-or-revise replanner output; never raises.

    Returns one of:
      {"kind": "keep", "rationale": str}
      {"kind": "revise", "axes": [...], "lenses": [...], "sources": str, "rationale": str}
      {"kind": "invalid", "reason": str}
    A revision restates every axis of the plan with its status in <axes>
    (at least ``axes_count``, numbered from 1): a revision cannot erase an
    unexplored dimension unstatused.
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
    axes_match = re.search(r"<axes>(.*?)</axes>", text, re.DOTALL)
    axes = numbered_list(axes_match.group(1)) if axes_match else None
    if axes is None or len(axes) < axes_count:
        return {
            "kind": "invalid",
            "reason": (
                "a revision restates every axis of the plan with its status "
                f"in <axes> ({axes_count} numbered from 1)"
            ),
        }
    rationale = re.search(
        r"<revision_rationale>(.*?)</revision_rationale>", text, re.DOTALL
    )
    sources = re.search(r"<sources>(.*?)</sources>", text, re.DOTALL)
    return {
        "kind": "revise",
        "axes": axes,
        "lenses": lenses,
        "sources": sources.group(1).strip() if sources else "",
        "rationale": (
            " ".join(rationale.group(1).split())
            if rationale and rationale.group(1).strip()
            else ""
        ),
    }


def member_roster_brief(ideation_ensemble: List[SessionRole]) -> str:
    """One line per ensemble member for the planner prompts."""
    return "\n".join(
        f"- member {i + 1}: cli={m.cli}, model={m.model}"
        + (
            " (has native web search during ideation)"
            if m.cli == "codex"
            else ""
        )
        for i, m in enumerate(ideation_ensemble)
    )


def run_lens_planner_session(
    prompt: str,
    ideation_dir: str,
    *,
    planner: SessionRole,
    bank_serving: Optional[Dict[str, str]] = None,
    env_strip: List[str],
    env_defaults: Dict[str, str],
    clamped_timeout: Callable[[Optional[float]], Optional[float]],
    artifacts_dir: str,
    staged: Optional[StagedHarness] = None,
):
    """One planner/replanner session on the planner's host; returns
    (result, cost_usd)."""
    from kapso.gated_mcp import get_mcp_config

    # The planner is the campaign's direction-setter, so it gets the bank
    # tools (serving-agentic-redesign.md §5): the index as a map of
    # directions, and the per-lens `bank:` declaration contract in its
    # prompt. Serving off -> bank gate unresolved -> plain session.
    mcp_servers, mcp_tools = get_mcp_config(
        gates=["bank"],
        include_base_tools=False,
        gate_failure_policy="skip",
        bank_serving=bank_serving,
        session=staged.session_id if staged is not None else None,
    )
    print(
        f"[GenericSearch] Lens planner starting: {planner.label}"
        f"{' (bank-served)' if mcp_servers else ''}"
    )
    own = {
            "env_strip": env_strip,
            "env_defaults": env_defaults,
            "mcp_servers": mcp_servers,
            "allowed_tools": [
                "Read",
                *(WEB_TOOLS if planner.web else []),
                *[t for t in mcp_tools if t.startswith("mcp__")],
                *(staged.allowed_tools if staged is not None else []),
            ],
            "disallowed_tools": [] if planner.web else WEB_TOOLS,
            "timeout": clamped_timeout(planner.timeout),
            "streaming": True,
            "planning_mode": False,
            "stream_artifact_path": codex_ideation.ideation_stream_path(
                artifacts_dir, "lens_planner",
                planner.model,
            ),
    }
    config = CodingAgentFactory.session_config(planner, own, staged)
    agent = CodingAgentFactory.create(config)
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
    ideation_lens_planner: Optional[SessionRole],
    ideation_ensemble: List[SessionRole],
    workspace_dir: str,
    iteration_count: int,
    shared_artifacts_brief: str,
    node_history: List[SearchNode],
    campaign_state_brief: Callable[[], str],
    get_best_experiment: Callable[[], Optional[SearchNode]],
    run_planner_session: Callable[[str, str], tuple],
    domain_guidance: str = "",
    experience_intro: str = "",
    campaign_notes: str = "",
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
        plan["axes"] = validated_axes(plan.get("axes"), plan_path)
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
                "domain_guidance": optional_block(domain_guidance),
                "experience_intro": optional_block(experience_intro),
                "campaign_notes": optional_block(campaign_notes),
            },
        )
        result, cost = run_planner_session(prompt, ideation_dir)
        if not result.success:
            raise RuntimeError(
                f"lens planner session failed: {result.error}"
            )
        parsed = parse_lens_plan(result.output, expected)
        plan = {
            "axes": parsed["axes"],
            "lenses": parsed["lenses"],
            "sources": parsed["sources"],
            "planner_model": planner.model,
            "iteration": iteration_count,
            "decision": "initial",
            "rationale": "",
        }
        write_lens_plan(plan_path, plan)
        append_lens_history(history_path, plan)
        for i, axis in enumerate(plan["axes"], 1):
            print(f"[GenericSearch] Axis {i}: {axis}")
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
            "axes": axes_brief(plan["axes"]),
            "campaign_state": campaign_state_brief(),
            "plan_iteration": str(plan.get("iteration", "?")),
            "previous_lenses": previous_lenses,
            "previous_sources": plan.get("sources") or "(none recorded)",
            "previous_rationale": plan.get("rationale") or "(none recorded)",
            "recent_feedback": recent_feedback,
            "champion_solution": champion_solution,
            "domain_guidance": optional_block(domain_guidance),
            "experience_intro": optional_block(experience_intro),
            "campaign_notes": optional_block(campaign_notes),
        },
    )
    result, cost = run_planner_session(prompt, ideation_dir)
    revision = (
        parse_lens_revision(result.output, expected, len(plan["axes"]))
        if result.success
        else {"kind": "invalid", "reason": f"session failed: {result.error}"}
    )
    if revision["kind"] == "revise":
        plan = {
            "axes": revision["axes"],
            "lenses": revision["lenses"],
            "sources": revision["sources"],
            "planner_model": planner.model,
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
                "axes": plan["axes"],
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
