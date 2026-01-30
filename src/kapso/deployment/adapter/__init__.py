# Deployment Adapter
#
# Adapts a solution repository for a specific deployment target.
# Uses coding agents (Aider, Claude Code, etc.) to transform the code.
#
# Usage:
#     from kapso.deployment.adapter import AdapterAgent
#     
#     adapter = AdapterAgent()
#     result = adapter.adapt(solution_path, goal, setting)

from kapso.deployment.adapter.agent import AdapterAgent
from kapso.deployment.adapter.validator import AdaptationValidator

__all__ = [
    "AdapterAgent",
    "AdaptationValidator",
]

