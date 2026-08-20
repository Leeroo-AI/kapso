"""K-way node-expansion lane configuration for the generic strategy.

Owns the expansion-lane feature: lane-count and per-lane env-overlay
validation, the lane-assignment prompt brief, the per-lane env overlay
lookup, and the round-representative pick. Stateless functions only —
GenericSearch assembles arguments from its state and delegates here.
"""

from typing import Any, Dict, List, Mapping, Optional, Tuple

from kapso.execution.search_strategies.base import SearchNode

MAX_NODE_EXPANSION = 8


def normalize_node_expansion(params: Mapping[str, Any]) -> Tuple[int, Optional[List[Dict[str, str]]]]:
    """Validate node_expansion_value and the optional per-lane env overlays."""
    raw = params.get("node_expansion_value", 1)
    if isinstance(raw, bool) or not isinstance(raw, int) or not 1 <= raw <= MAX_NODE_EXPANSION:
        raise ValueError(
            f"node_expansion_value must be an int in [1, {MAX_NODE_EXPANSION}]"
        )
    lane_env = params.get("expansion_lane_env")
    if lane_env is not None:
        if not isinstance(lane_env, list) or not all(
            isinstance(e, dict) and all(
                isinstance(k, str) and isinstance(v, str) for k, v in e.items()
            )
            for e in lane_env
        ):
            raise ValueError(
                "expansion_lane_env must be a list of {str: str} mappings "
                "(one per lane, e.g. CUDA_VISIBLE_DEVICES pins)"
            )
    return raw, lane_env


def validate_node_expansion_config(
    expansion: int,
    ensemble: Optional[List[Dict[str, str]]],
    selector: Optional[Dict[str, str]],
) -> None:
    """K>1 requires the ensemble+selector flow (the selector emits the K)."""
    if expansion > 1 and (not ensemble or selector is None):
        raise ValueError(
            "node_expansion_value > 1 requires ideation_ensemble and "
            "ideation_selector (the selector emits the top-K solutions)"
        )


def render_lane_brief(
    lane_index: int,
    lane_count: int,
    lane_env: Optional[Mapping[str, str]],
) -> str:
    """Prompt block announcing this lane's env assignment.

    An env pin is a fence the agent cannot see: the session inherits it,
    but hardware probes (nvidia-smi) ignore it and per-command exports
    silently override it — the first K=2 flight had a lane discover the
    "idle" sibling GPU that way. Empty/absent lane env renders nothing.
    """
    if not lane_env:
        return ""
    pins = "\n".join(f"- `{key}={value}`" for key, value in lane_env.items())
    if lane_count > 1:
        return (
            "## Parallel Lane Assignment (read first)\n\n"
            f"You are implementation lane {lane_index} of {lane_count} — "
            f"{lane_count - 1} sibling lane(s) are running CONCURRENTLY on "
            "this machine, implementing different solutions on their own "
            "branches.\n\n"
            "Your session environment carries these lane-exclusive "
            "overrides:\n"
            f"{pins}\n\n"
            "Each sibling lane received DIFFERENT values for the same "
            "variables — they partition this machine's resources between "
            "lanes. Treat yours as an exclusive assignment:\n"
            "- Do NOT override, unset, or widen these variables in your own "
            "commands; plain commands already inherit them.\n"
            "- Do NOT claim resources outside your assignment even if they "
            "look idle — hardware probes (e.g. `nvidia-smi`) list ALL "
            "physical devices, including your siblings'.\n"
            "- Task-level shared directories (artifacts, submission) are "
            "visible to every lane: namespace the files you create and "
            "follow the task's promotion protocol exactly."
        )
    return (
        "## Session Environment Pins\n\n"
        "The orchestrator set these run-level environment overrides for "
        "this session:\n"
        f"{pins}\n\n"
        "Do not override or unset them in your commands; plain commands "
        "already inherit them."
    )


def lane_env_overlay(
    expansion_lane_env: Optional[List[Dict[str, str]]],
    lane_index: int,
) -> Optional[Dict[str, str]]:
    """This lane's env pin mapping (None when no overlay is configured)."""
    return (
        expansion_lane_env[lane_index]
        if expansion_lane_env
        and lane_index < len(expansion_lane_env)
        else None
    )


def pick_representative(
    nodes: List[SearchNode], maximize_scoring: bool
) -> SearchNode:
    """Best-scoring node of the round; scoreless nodes rank last."""
    if len(nodes) == 1:
        return nodes[0]

    def sort_key(node: SearchNode):
        if node.score is None:
            return (0, 0.0)
        return (
            1,
            node.score
            if maximize_scoring
            else -node.score,
        )
    return max(nodes, key=sort_key)
