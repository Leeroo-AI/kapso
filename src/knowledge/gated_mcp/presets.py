"""
Preset configurations for the Gated MCP Server.

Presets define which gates are enabled and their configuration parameters.
Each preset is designed for a specific use case (merger, ideation, etc.).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GateConfig:
    """Configuration for a single gate."""
    
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Preset:
    """A named preset configuration."""
    
    name: str
    description: str
    gates: Dict[str, GateConfig]


# =============================================================================
# Gate Tool Names Mapping
# =============================================================================

# Maps gate names to their tool names (for allowed_tools generation)
GATE_TOOL_NAMES: Dict[str, List[str]] = {
    "kg": [
        "search_knowledge",
        "get_wiki_page",
        "kg_index",
        "kg_edit",
        "get_page_structure",
    ],
    "idea": [
        "wiki_idea_search",
    ],
    "code": [
        "wiki_code_search",
    ],
    "research": [
        "research_idea",
        "research_implementation",
        "research_study",
    ],
}


# =============================================================================
# Preset Definitions
# =============================================================================

PRESETS: Dict[str, Preset] = {
    # For KnowledgeMerger - needs full KG access
    "merger": Preset(
        name="merger",
        description="Full KG access for KnowledgeMerger",
        gates={
            "kg": GateConfig(enabled=True, params={"include_content": True}),
        },
    ),
    
    # For ideation phase - needs idea search + research
    "ideation": Preset(
        name="ideation",
        description="Idea search + web research for ideation phase",
        gates={
            "idea": GateConfig(enabled=True, params={
                "top_k": 10,
                "use_llm_reranker": True,
                "include_content": True,
            }),
            "research": GateConfig(enabled=True, params={
                "default_depth": "deep",
                "default_top_k": 5,
            }),
        },
    ),
    
    # For implementation phase - needs code search + research
    "implementation": Preset(
        name="implementation",
        description="Code search + web research for implementation phase",
        gates={
            "code": GateConfig(enabled=True, params={
                "top_k": 5,
                "use_llm_reranker": True,
                "include_content": True,
            }),
            "research": GateConfig(enabled=True, params={
                "default_depth": "deep",
                "default_top_k": 3,
            }),
        },
    ),
    
    # For context managers - read-only search (faster, no reranking)
    "context": Preset(
        name="context",
        description="Fast read-only search for context managers",
        gates={
            "idea": GateConfig(enabled=True, params={
                "top_k": 5,
                "use_llm_reranker": False,
                "include_content": False,
            }),
            "code": GateConfig(enabled=True, params={
                "top_k": 5,
                "use_llm_reranker": False,
                "include_content": False,
            }),
        },
    ),
    
    # All tools enabled with default settings (for debugging/admin)
    "full": Preset(
        name="full",
        description="All gates enabled with default settings",
        gates={
            "kg": GateConfig(enabled=True),
            "idea": GateConfig(enabled=True),
            "code": GateConfig(enabled=True),
            "research": GateConfig(enabled=True),
        },
    ),
}


# =============================================================================
# Helper Functions
# =============================================================================

def get_preset(name: str) -> Preset:
    """
    Get a preset by name.
    
    Args:
        name: Preset name (merger, ideation, implementation, context, full)
        
    Returns:
        Preset configuration
        
    Raises:
        ValueError: If preset name is unknown
    """
    if name not in PRESETS:
        available = ", ".join(PRESETS.keys())
        raise ValueError(f"Unknown preset: '{name}'. Available: {available}")
    return PRESETS[name]


def get_allowed_tools_for_preset(
    preset_name: str,
    mcp_server_name: str,
    include_base_tools: bool = True,
) -> List[str]:
    """
    Generate the allowed_tools list for Claude Code based on a preset.
    
    This function returns the full list of MCP tool names that should be
    allowed for a given preset, formatted for Claude Code's allowed_tools.
    
    Args:
        preset_name: Name of the preset
        mcp_server_name: Name of the MCP server (e.g., "gated-knowledge")
        include_base_tools: Include base tools like Read, Write, Bash (default True)
        
    Returns:
        List of tool names for allowed_tools config
        
    Example:
        >>> get_allowed_tools_for_preset("ideation", "gated-knowledge")
        ["Read", "Write", "Bash", "mcp__gated-knowledge__wiki_idea_search", ...]
    """
    preset = get_preset(preset_name)
    
    tools: List[str] = []
    
    # Add base tools if requested
    if include_base_tools:
        tools.extend(["Read", "Write", "Bash"])
    
    # Add MCP tools for each enabled gate
    for gate_name, gate_config in preset.gates.items():
        if gate_config.enabled and gate_name in GATE_TOOL_NAMES:
            for tool_name in GATE_TOOL_NAMES[gate_name]:
                # Format: mcp__<server>__<tool>
                mcp_tool = f"mcp__{mcp_server_name}__{tool_name}"
                tools.append(mcp_tool)
    
    return tools


def list_presets() -> List[str]:
    """Return list of available preset names."""
    return list(PRESETS.keys())
