"""
Gate definitions and configuration for the Gated MCP Server.

Each gate groups related tools with default configuration parameters.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class GateDefinition:
    """Definition of a gate with its tools and default config."""
    
    tools: List[str]
    default_params: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Gate Definitions
# =============================================================================

GATES: Dict[str, GateDefinition] = {
    "kg": GateDefinition(
        tools=[
            "search_knowledge",
            "get_wiki_page",
            "kg_index",
            "kg_edit",
            "get_page_structure",
        ],
        default_params={"include_content": True},
    ),
    "idea": GateDefinition(
        tools=["wiki_idea_search"],
        default_params={
            "top_k": 5,
            "use_llm_reranker": True,
            "include_content": True,
        },
    ),
    "code": GateDefinition(
        tools=["wiki_code_search"],
        default_params={
            "top_k": 5,
            "use_llm_reranker": True,
            "include_content": True,
        },
    ),
    "research": GateDefinition(
        tools=[
            "research_idea",
            "research_implementation",
            "research_study",
        ],
        default_params={
            "default_depth": "deep",
            "default_top_k": 5,
        },
    ),
}


# =============================================================================
# Helper Functions
# =============================================================================

def get_allowed_tools_for_gates(
    gates: List[str],
    mcp_server_name: str,
    include_base_tools: bool = True,
) -> List[str]:
    """
    Generate the allowed_tools list for Claude Code based on gate names.
    
    Args:
        gates: List of gate names (e.g., ["idea", "research"])
        mcp_server_name: Name of the MCP server (e.g., "gated-knowledge")
        include_base_tools: Include base tools like Read, Write, Bash (default True)
        
    Returns:
        List of tool names for allowed_tools config
        
    Example:
        >>> get_allowed_tools_for_gates(["idea", "research"], "gated-knowledge")
        ["Read", "Write", "Bash", "mcp__gated-knowledge__wiki_idea_search", ...]
    """
    tools: List[str] = []
    
    # Add base tools if requested
    if include_base_tools:
        tools.extend(["Read", "Write", "Bash"])
    
    # Add MCP tools for each gate
    for gate_name in gates:
        if gate_name in GATES:
            for tool_name in GATES[gate_name].tools:
                # Format: mcp__<server>__<tool>
                mcp_tool = f"mcp__{mcp_server_name}__{tool_name}"
                tools.append(mcp_tool)
    
    return tools


def list_gates() -> List[str]:
    """Return list of available gate names."""
    return list(GATES.keys())


def get_gate_config(gate_name: str) -> GateDefinition:
    """
    Get a gate definition by name.
    
    Args:
        gate_name: Gate name (kg, idea, code, research)
        
    Returns:
        GateDefinition with tools and default_params
        
    Raises:
        ValueError: If gate name is unknown
    """
    if gate_name not in GATES:
        available = ", ".join(GATES.keys())
        raise ValueError(f"Unknown gate: '{gate_name}'. Available: {available}")
    return GATES[gate_name]
