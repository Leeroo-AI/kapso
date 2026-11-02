"""Fallback technical-difficulties generator.

The implementor is the primary author of ``<technical_difficulties>``
(its output contract). When a session crashed, was deadline-killed, or
simply omitted the tag, this module reconstructs the report post-hoc: a
short read-only session on the implementation role's host over the
implementor's persisted stream artifact and the workspace, emitting the
same tag.

The trigger is purely mechanical — the field is missing — never score-
or outcome-based.
"""

import os
import re
import time
from typing import Dict, Optional, Tuple

from kapso.core.prompt_loader import load_prompt, render_prompt
from kapso.execution.coding_agents.factory import CodingAgentFactory
from kapso.core.roles import SessionRole
from kapso.harness.staging import StagedHarness

_TAG_PATTERN = re.compile(
    r"<technical_difficulties>\s*(.*?)\s*</technical_difficulties>",
    re.DOTALL,
)


def generate_technical_difficulties(
    role: SessionRole,
    env_strip: list,
    timeout_seconds,
    workspace_dir: str,
    solution: str,
    stream_artifact_path: str,
    branch_name: str,
    parent_branch_name: str,
    staged: Optional[StagedHarness],
) -> Tuple[str, Dict[str, float]]:
    """Reconstruct the technical-difficulties report for one session, on
    the implementation role's host, staged and recorded like every session
    (`difficulties/<branch>`); its prompt reads the branch before it
    concludes what was lost.

    Returns the report text ("" if the fallback session itself failed —
    the node then carries an empty field, which downstream extraction
    treats as "(none reported)") and the session's telemetry, which is
    paid for either way.
    """
    template = load_prompt(
        "execution/search_strategies/generic/prompts/difficulties_fallback.md"
    )
    prompt = render_prompt(
        template,
        {
            "solution": solution,
            "branch": branch_name,
            "parent_branch": parent_branch_name,
            "stream_artifact_path": (
                stream_artifact_path
                if os.path.isfile(stream_artifact_path)
                else "(no stream artifact was persisted)"
            ),
        },
    )

    config = CodingAgentFactory.session_config(role, {
        "env_strip": env_strip,
        "allowed_tools": ["Read", "Bash"],
        "sandbox": "read-only",
        "timeout": timeout_seconds,
        "streaming": True,
        "planning_mode": False,
    }, staged)
    agent = CodingAgentFactory.create(config)
    agent.initialize(workspace_dir)
    started = time.monotonic()
    result = agent.generate_code(prompt)
    telemetry = {
        "cost_usd": agent.get_cumulative_cost(),
        "duration_seconds": time.monotonic() - started,
    }
    agent.cleanup()

    if not result.success or not result.output:
        return "", telemetry
    match = _TAG_PATTERN.search(result.output)
    if match:
        return match.group(1).strip(), telemetry
    return result.output.strip(), telemetry
