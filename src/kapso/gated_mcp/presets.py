"""
Gate definitions and configuration for the Gated MCP Server.

Each gate groups related tools with default configuration parameters.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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
    "experiment_history": GateDefinition(
        tools=[
            "get_top_experiments",
            "get_recent_experiments",
            "search_similar_experiments",
            "get_insights",
        ],
        default_params={
            "top_k": 5,
            "recent_k": 5,
            "similar_k": 3,
        },
    ),
    "repo_memory": GateDefinition(
        tools=[
            "get_repo_memory_section",
            "list_repo_memory_sections",
            "get_repo_memory_summary",
        ],
        default_params={},
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


def get_mcp_config(
    gates: List[str],
    server_name: str = "gated-knowledge",
    project_root: Optional[Path] = None,
    kg_index_path: Optional[str] = None,
    experiment_history_path: Optional[str] = None,
    weaviate_url: Optional[str] = None,
    repo_root: Optional[str] = None,
    include_base_tools: bool = True,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Get MCP server config and allowed tools for the given gates.
    
    Args:
        gates: List of gate names (e.g., ["idea", "research", "experiment_history"])
        server_name: MCP server name (default: "gated-knowledge")
        project_root: Project root path (defaults to 2 levels up from this file)
        kg_index_path: Path to .index file. Required if "kg", "idea", or "code" 
                       gates are enabled. Falls back to KG_INDEX_PATH env var.
        experiment_history_path: Path to experiment history JSON file. Required if
                                 "experiment_history" gate is enabled.
        weaviate_url: Weaviate URL for semantic search (optional).
        repo_root: Path to repo root for repo_memory gate. Falls back to 
                   REPO_MEMORY_ROOT env var or CWD.
        include_base_tools: Include Read, Write, Bash in allowed_tools (default True)
    
    Returns:
        Tuple of (mcp_servers dict, allowed_tools list)
        
    Example:
        >>> mcp_servers, allowed_tools = get_mcp_config(["idea", "research"])
        >>> # Use in Claude Code config:
        >>> config = CodingAgentConfig(agent_specific={
        ...     "mcp_servers": mcp_servers,
        ...     "allowed_tools": allowed_tools,
        ... })
    """
    # Resolve project root
    if project_root is None:
        # Default: 3 levels up from this file (src/knowledge/gated_mcp -> project root)
        project_root = Path(__file__).parent.parent.parent.parent
    
    # Build environment for MCP server
    mcp_env: Dict[str, str] = {
        "PYTHONPATH": str(project_root),
        "MCP_ENABLED_GATES": ",".join(gates),
    }
    
    # Resolve kg_index_path (needed for kg, idea, code gates)
    kg_gates = {"kg", "idea", "code"}
    needs_kg = bool(kg_gates & set(gates))
    
    if needs_kg:
        resolved_kg_path = kg_index_path or os.environ.get("KG_INDEX_PATH")
        if resolved_kg_path:
            mcp_env["KG_INDEX_PATH"] = resolved_kg_path
    
    # Resolve experiment_history_path (needed for experiment_history gate)
    if "experiment_history" in gates:
        resolved_history_path = experiment_history_path or os.environ.get("EXPERIMENT_HISTORY_PATH")
        if resolved_history_path:
            mcp_env["EXPERIMENT_HISTORY_PATH"] = resolved_history_path
        
        # Add Weaviate URL if available
        resolved_weaviate_url = weaviate_url or os.environ.get("WEAVIATE_URL")
        if resolved_weaviate_url:
            mcp_env["WEAVIATE_URL"] = resolved_weaviate_url
    
    # Resolve repo_root (needed for repo_memory gate)
    if "repo_memory" in gates:
        resolved_repo_root = repo_root or os.environ.get("REPO_MEMORY_ROOT")
        if resolved_repo_root:
            mcp_env["REPO_MEMORY_ROOT"] = resolved_repo_root
    
    # Build MCP servers config
    mcp_servers = {
        server_name: {
            "command": "python",
            "args": ["-m", "kapso.gated_mcp.server"],
            "cwd": str(project_root),
            "env": mcp_env,
        }
    }
    
    # Get allowed tools
    allowed_tools = get_allowed_tools_for_gates(
        gates, server_name, include_base_tools=include_base_tools
    )
    
    return mcp_servers, allowed_tools


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
