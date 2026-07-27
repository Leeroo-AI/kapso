"""Single structural authority for coding-agent CLI capabilities."""

from __future__ import annotations

from typing import Final

_SUPPORTED_EFFORTS: Final = {
    "codex": frozenset({"minimal", "low", "medium", "high", "xhigh"}),
    "claude_code": frozenset({"low", "medium", "high", "xhigh", "max"}),
}
_READ_ONLY_TOOLS: Final = {
    "codex": frozenset({"Read", "WebSearch"}),
    "claude_code": frozenset({"Glob", "Grep", "Read", "WebSearch"}),
}
_EDIT_TOOLS: Final = {
    "codex": frozenset(),
    "claude_code": frozenset({"Edit", "Write"}),
}


def coding_agent_supported_efforts(cli: str) -> frozenset[str]:
    """Return the closed effort vocabulary for one supported CLI."""

    if cli not in _SUPPORTED_EFFORTS:
        raise ValueError("coding-agent CLI is unsupported")
    return _SUPPORTED_EFFORTS[cli]


def coding_agent_supported_tools(
    cli: str,
    *,
    edit_workspace: bool,
) -> frozenset[str]:
    """Return tools compatible with the CLI and physical workspace authority."""

    if cli not in _READ_ONLY_TOOLS or type(edit_workspace) is not bool:
        raise ValueError("coding-agent CLI tool query is invalid")
    tools = _READ_ONLY_TOOLS[cli]
    return tools | _EDIT_TOOLS[cli] if edit_workspace else tools


__all__ = [
    "coding_agent_supported_efforts",
    "coding_agent_supported_tools",
]
