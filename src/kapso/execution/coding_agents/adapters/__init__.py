# Coding Agent Adapters
#
# Each adapter wraps a specific coding tool/agent to conform to
# the CodingAgentInterface.
#
# Available adapters:
# - aider_agent.py: Aider (native git support)
# - gemini_agent.py: Gemini CLI (Google GenAI SDK)
# - claude_code_agent.py: Claude Code (Anthropic CLI)
# - openhands_agent.py: OpenHands (Docker sandbox)
#
# NOTE: Adapters are imported lazily by the factory to avoid import errors
# when dependencies aren't installed. Don't import them here directly.

__all__ = [
    "AiderCodingAgent",
    "GeminiCodingAgent",
    "ClaudeCodeCodingAgent",
    "OpenHandsCodingAgent",
]


def __getattr__(name):
    """Lazy import adapters to avoid dependency errors."""
    if name == "AiderCodingAgent":
        from kapso.execution.coding_agents.adapters.aider_agent import AiderCodingAgent
        return AiderCodingAgent
    elif name == "GeminiCodingAgent":
        from kapso.execution.coding_agents.adapters.gemini_agent import GeminiCodingAgent
        return GeminiCodingAgent
    elif name == "ClaudeCodeCodingAgent":
        from kapso.execution.coding_agents.adapters.claude_code_agent import ClaudeCodeCodingAgent
        return ClaudeCodeCodingAgent
    elif name == "OpenHandsCodingAgent":
        from kapso.execution.coding_agents.adapters.openhands_agent import OpenHandsCodingAgent
        return OpenHandsCodingAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

