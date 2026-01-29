"""Gate implementations for the Gated MCP Server."""

from src.knowledge.gated_mcp.gates.base import ToolGate
from src.knowledge.gated_mcp.gates.kg_gate import KGGate
from src.knowledge.gated_mcp.gates.idea_gate import IdeaGate
from src.knowledge.gated_mcp.gates.code_gate import CodeGate
from src.knowledge.gated_mcp.gates.research_gate import ResearchGate

__all__ = [
    "ToolGate",
    "KGGate",
    "IdeaGate",
    "CodeGate",
    "ResearchGate",
]
