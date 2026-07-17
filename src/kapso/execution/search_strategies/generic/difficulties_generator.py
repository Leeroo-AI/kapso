"""Fallback technical-difficulties generator.

The implementor is the primary author of ``<technical_difficulties>``
(its output contract). When a session crashed, was deadline-killed, or
simply omitted the tag, this module reconstructs the report post-hoc: a
short read-only Claude session over the implementor's persisted stream
artifact and the workspace, emitting the same tag.

The trigger is purely mechanical — the field is missing — never score-
or outcome-based.
"""

import os
import re

from kapso.core.prompt_loader import load_prompt, render_prompt
from kapso.execution.coding_agents.base import CodingAgentConfig
from kapso.execution.coding_agents.adapters.claude_code_agent import (
    ClaudeCodeCodingAgent,
)

_TAG_PATTERN = re.compile(
    r"<technical_difficulties>\s*(.*?)\s*</technical_difficulties>",
    re.DOTALL,
)


def generate_technical_difficulties(
    model: str,
    claude_auth_settings: dict,
    aws_region: str,
    env_strip: list,
    effort,
    timeout_seconds: float,
    workspace_dir: str,
    solution: str,
    stream_artifact_path: str,
) -> str:
    """Reconstruct the technical-difficulties report for one session.

    Returns the report text ("" if the fallback session itself failed —
    the node then carries an empty field, which downstream extraction
    treats as "(none reported)").
    """
    template = load_prompt(
        "execution/search_strategies/generic/prompts/difficulties_fallback.md"
    )
    prompt = render_prompt(
        template,
        {
            "solution": solution,
            "stream_artifact_path": (
                stream_artifact_path
                if os.path.isfile(stream_artifact_path)
                else "(no stream artifact was persisted)"
            ),
        },
    )

    config = CodingAgentConfig(
        agent_type="claude_code",
        model=model,
        debug_model=model,
        agent_specific={
            **claude_auth_settings,
            "env_strip": env_strip,
            "aws_region": aws_region,
            "allowed_tools": ["Read", "Bash"],
            "timeout": timeout_seconds,
            "streaming": True,
            "planning_mode": False,
            "effort": effort,
        },
    )
    agent = ClaudeCodeCodingAgent(config)
    agent.initialize(workspace_dir)
    result = agent.generate_code(prompt)
    agent.cleanup()

    if not result.success or not result.output:
        return ""
    match = _TAG_PATTERN.search(result.output)
    if match:
        return match.group(1).strip()
    return result.output.strip()
