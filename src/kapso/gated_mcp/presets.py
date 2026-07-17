"""
Gate definitions and configuration for the Gated MCP Server.

Each gate groups related tools with default configuration parameters.
"""

import logging
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

logger = logging.getLogger(__name__)


@dataclass
class GateDefinition:
    """Definition of a gate with its tools and default config.
    
    For internal gates (bundled into gated-knowledge server), server_name and
    command are None. For external gates (e.g., leeroopedia-mcp), set server_name
    to the MCP server name and command to the CLI entry point.
    """
    
    tools: List[str]
    default_params: Dict[str, Any] = field(default_factory=dict)
    # External server fields (None = bundled in gated-knowledge)
    server_name: Optional[str] = None
    command: Optional[str] = None
    required_env: List[str] = field(default_factory=list)
    required_commands: List[str] = field(default_factory=list)
    # Deprecated construction alias retained for downstream registries.
    env_keys: List[str] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self.required_env = list(
            dict.fromkeys([*self.required_env, *self.env_keys])
        )
        self.env_keys = list(self.required_env)


@dataclass(frozen=True)
class GateDiagnostic:
    """Capability check result for one requested gate."""

    gate_name: str
    enabled: bool
    missing_env: Tuple[str, ...] = ()
    missing_commands: Tuple[str, ...] = ()

    @property
    def reason(self) -> str:
        if self.enabled:
            return "available"

        parts = []
        if self.missing_env:
            parts.append(f"missing environment: {', '.join(self.missing_env)}")
        if self.missing_commands:
            parts.append(f"missing commands: {', '.join(self.missing_commands)}")
        return "; ".join(parts) or "unavailable"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate_name": self.gate_name,
            "enabled": self.enabled,
            "reason": self.reason,
            "missing_env": list(self.missing_env),
            "missing_commands": list(self.missing_commands),
        }


@dataclass(frozen=True)
class GateResolution:
    """Resolved gates plus a diagnostic for every requested gate."""

    requested_gates: Tuple[str, ...]
    enabled_gates: Tuple[str, ...]
    diagnostics: Tuple[GateDiagnostic, ...]

    @property
    def unavailable_gates(self) -> Tuple[str, ...]:
        return tuple(
            diagnostic.gate_name
            for diagnostic in self.diagnostics
            if not diagnostic.enabled
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "requested_gates": list(self.requested_gates),
            "enabled_gates": list(self.enabled_gates),
            "unavailable_gates": list(self.unavailable_gates),
            "diagnostics": [
                diagnostic.to_dict() for diagnostic in self.diagnostics
            ],
        }


class GateCapabilityError(RuntimeError):
    """Raised when required gate capabilities are unavailable in error mode."""

    def __init__(self, diagnostics: Sequence[GateDiagnostic]):
        self.diagnostics = tuple(diagnostics)
        details = "; ".join(
            f"{diagnostic.gate_name}: {diagnostic.reason}"
            for diagnostic in self.diagnostics
        )
        super().__init__(f"Gate capability requirements not met: {details}")


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
        required_env=["KG_INDEX_PATH"],
    ),
    "idea": GateDefinition(
        tools=["wiki_idea_search"],
        default_params={
            "top_k": 5,
            "use_llm_reranker": True,
            "include_content": True,
        },
        required_env=["KG_INDEX_PATH"],
    ),
    "code": GateDefinition(
        tools=["wiki_code_search"],
        default_params={
            "top_k": 5,
            "use_llm_reranker": True,
            "include_content": True,
        },
        required_env=["KG_INDEX_PATH"],
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
        required_env=["OPENAI_API_KEY"],
    ),
    "experiment_history": GateDefinition(
        tools=[
            "get_top_experiments",
            "get_recent_experiments",
            "search_similar_experiments",
        ],
        default_params={
            "top_k": 5,
            "recent_k": 5,
            "similar_k": 3,
        },
        required_env=["EXPERIMENT_HISTORY_PATH"],
    ),
    "repo_memory": GateDefinition(
        tools=[
            "get_repo_memory_section",
            "list_repo_memory_sections",
            "get_repo_memory_summary",
        ],
        default_params={},
    ),
    # External MCP server: leeroopedia-mcp (api.leeroopedia.com)
    # Runs as a separate process, not bundled in gated-knowledge
    "leeroopedia": GateDefinition(
        tools=[
            "search_knowledge",
            "build_plan",
            "review_plan",
            "verify_code_math",
            "diagnose_failure",
            "propose_hypothesis",
            "query_hyperparameter_priors",
            "get_page",
        ],
        default_params={},
        server_name="leeroopedia",
        command="leeroopedia-mcp",
        required_env=["LEEROOPEDIA_API_KEY"],
        required_commands=["leeroopedia-mcp"],
    ),
}


# =============================================================================
# Helper Functions
# =============================================================================

GATE_FAILURE_POLICIES = frozenset({"skip", "warn", "error"})


def _normalize_gate_names(gates: Sequence[str]) -> Tuple[str, ...]:
    if isinstance(gates, str):
        raise TypeError("gates must be a sequence of gate names, not a string")

    normalized = []
    for gate in gates:
        name = str(gate).strip()
        if name and name not in normalized:
            normalized.append(name)

    unknown = [name for name in normalized if name not in GATES]
    if unknown:
        available = ", ".join(GATES)
        requested = ", ".join(unknown)
        raise ValueError(
            f"Unknown gate(s): {requested}. Available gates: {available}"
        )
    return tuple(normalized)


def resolve_gates(
    gates: Sequence[str],
    *,
    policy: str = "warn",
    env: Optional[Mapping[str, str]] = None,
    command_resolver: Optional[Callable[[str], Optional[str]]] = None,
) -> GateResolution:
    """Resolve requested gates against their declared capabilities.

    ``skip`` and ``warn`` both omit unavailable gates; ``warn`` additionally
    logs a diagnostic. ``error`` raises one aggregate ``GateCapabilityError``.
    Unknown gates and policies are always configuration errors.
    """
    normalized_policy = str(policy).strip().lower()
    if normalized_policy not in GATE_FAILURE_POLICIES:
        choices = ", ".join(sorted(GATE_FAILURE_POLICIES))
        raise ValueError(
            f"Invalid gate failure policy {policy!r}. Expected one of: {choices}"
        )

    requested = _normalize_gate_names(gates)
    effective_env = os.environ if env is None else env
    resolve_command = command_resolver or shutil.which
    diagnostics = []

    for gate_name in requested:
        definition = GATES[gate_name]
        required_commands = list(definition.required_commands)
        if definition.command:
            required_commands.append(definition.command)
        required_commands = list(dict.fromkeys(required_commands))

        missing_env = tuple(
            name for name in definition.required_env if not effective_env.get(name)
        )
        missing_commands = tuple(
            command
            for command in required_commands
            if not resolve_command(command)
        )
        diagnostics.append(
            GateDiagnostic(
                gate_name=gate_name,
                enabled=not missing_env and not missing_commands,
                missing_env=missing_env,
                missing_commands=missing_commands,
            )
        )

    unavailable = [item for item in diagnostics if not item.enabled]
    if unavailable and normalized_policy == "error":
        raise GateCapabilityError(unavailable)
    if normalized_policy == "warn":
        for diagnostic in unavailable:
            logger.warning(
                "Skipping unavailable MCP gate '%s': %s",
                diagnostic.gate_name,
                diagnostic.reason,
            )

    return GateResolution(
        requested_gates=requested,
        enabled_gates=tuple(
            item.gate_name for item in diagnostics if item.enabled
        ),
        diagnostics=tuple(diagnostics),
    )


def get_allowed_tools_for_gates(
    gates: Sequence[str],
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
    gate_names = _normalize_gate_names(gates)
    tools: List[str] = []
    
    # Add base tools if requested
    if include_base_tools:
        tools.extend(["Read", "Write", "Bash"])
    
    # Add MCP tools for each gate
    # External gates (with server_name set) use their own server name prefix
    for gate_name in gate_names:
        gate_def = GATES[gate_name]
        effective_server = gate_def.server_name or mcp_server_name
        for tool_name in gate_def.tools:
            # Format: mcp__<server>__<tool>
            mcp_tool = f"mcp__{effective_server}__{tool_name}"
            tools.append(mcp_tool)
    
    return tools


def get_mcp_config(
    gates: Sequence[str],
    server_name: str = "gated-knowledge",
    project_root: Optional[Path] = None,
    kg_index_path: Optional[str] = None,
    experiment_history_path: Optional[str] = None,
    weaviate_url: Optional[str] = None,
    repo_root: Optional[str] = None,
    include_base_tools: bool = True,
    gate_failure_policy: str = "warn",
    command_resolver: Optional[Callable[[str], Optional[str]]] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Get MCP server config and allowed tools for the given gates.
    
    Args:
        gates: List of gate names (e.g., ["idea", "research", "experiment_history"])
        server_name: MCP server name (default: "gated-knowledge")
        project_root: Project root path (defaults to the Kapso checkout root)
        kg_index_path: Path to .index file. Required if "kg", "idea", or "code" 
                       gates are enabled. Falls back to KG_INDEX_PATH env var.
        experiment_history_path: Path to experiment history JSON file. Required if
                                 "experiment_history" gate is enabled.
        weaviate_url: Weaviate URL for semantic search (optional).
        repo_root: Path to repo root for repo_memory gate. Falls back to 
                   REPO_MEMORY_ROOT env var or CWD.
        include_base_tools: Include Read, Write, Bash in allowed_tools (default True)
        gate_failure_policy: Missing-capability policy: skip, warn, or error.
        command_resolver: Optional command lookup override for testing.
    
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
    # Explicit path arguments behave like environment capabilities for gate
    # resolution, without mutating the caller's process environment.
    effective_env = dict(os.environ)
    explicit_env = {
        "KG_INDEX_PATH": kg_index_path,
        "EXPERIMENT_HISTORY_PATH": experiment_history_path,
        "WEAVIATE_URL": weaviate_url,
        "REPO_MEMORY_ROOT": repo_root,
    }
    effective_env.update(
        {key: str(value) for key, value in explicit_env.items() if value}
    )

    resolution = resolve_gates(
        gates,
        policy=gate_failure_policy,
        env=effective_env,
        command_resolver=command_resolver,
    )
    enabled_gates = list(resolution.enabled_gates)

    # Resolve project root
    if project_root is None:
        # src/kapso/gated_mcp/presets.py -> checkout root
        project_root = Path(__file__).parent.parent.parent.parent
    project_root = Path(project_root).expanduser().resolve()
    python_path = project_root / "src"
    if not python_path.is_dir():
        python_path = project_root
    
    # Split gates into internal (bundled in gated-knowledge) and external (separate servers)
    internal_gates = [
        gate_name
        for gate_name in enabled_gates
        if GATES[gate_name].command is None
    ]
    
    # Build environment for MCP server (internal gates only)
    mcp_env: Dict[str, str] = {
        "PYTHONPATH": str(python_path),
        "MCP_ENABLED_GATES": ",".join(internal_gates),
        "MCP_GATE_FAILURE_POLICY": "error",
    }

    # Forward required environment plus optional per-gate context only for
    # enabled internal gates.
    for gate_name in internal_gates:
        for key in GATES[gate_name].required_env:
            mcp_env[key] = effective_env[key]
    if "experiment_history" in internal_gates and effective_env.get("WEAVIATE_URL"):
        mcp_env["WEAVIATE_URL"] = effective_env["WEAVIATE_URL"]
    if "repo_memory" in internal_gates and effective_env.get("REPO_MEMORY_ROOT"):
        mcp_env["REPO_MEMORY_ROOT"] = effective_env["REPO_MEMORY_ROOT"]
    
    # Build MCP servers config (gated-knowledge for internal gates)
    mcp_servers: Dict[str, Any] = {}
    if internal_gates:
        mcp_servers[server_name] = {
            "command": sys.executable,
            "args": ["-m", "kapso.gated_mcp.server"],
            "cwd": str(project_root),
            "env": mcp_env,
        }
    
    # Add external MCP servers (e.g., leeroopedia-mcp)
    for gate_name in enabled_gates:
        gate_def = GATES[gate_name]
        if gate_def.command and gate_def.server_name:
            ext_env = {}
            for key in gate_def.required_env:
                val = effective_env.get(key, "")
                if val:
                    ext_env[key] = val
            mcp_servers[gate_def.server_name] = {
                "command": gate_def.command,
                "env": ext_env,
            }
    
    # Get allowed tools
    allowed_tools = get_allowed_tools_for_gates(
        enabled_gates, server_name, include_base_tools=include_base_tools
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
