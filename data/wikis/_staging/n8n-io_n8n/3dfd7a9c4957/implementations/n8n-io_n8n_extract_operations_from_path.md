# Implementation: _extract_operations_from_path

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
|-
! Domains
| [[domain::Graph_Theory]], [[domain::Workflow_Evaluation]]
|-
! Last Updated
| [[last_updated::2024-12-18 12:00 GMT]]
|}

== Overview ==

Concrete function for converting NetworkX edit paths into human-readable edit operation dictionaries.

=== Description ===

`_extract_operations_from_path()` processes the raw edit paths from GED:

1. **Node Edit Processing**: Loops through `(u, v)` tuples in node_edit_path
   - `(None, v)`: Node insertion - creates "Add missing node" description
   - `(u, None)`: Node deletion - creates "Remove node" description
   - `(u, v)`: Node substitution - creates type change or parameter update description

2. **Edge Edit Processing**: Loops through `(e1, e2)` tuples in edge_edit_path
   - `(None, e2)`: Edge insertion - creates "Add connection" description
   - `(e1, None)`: Edge deletion - creates "Remove connection" description
   - `(e1, e2)`: Edge substitution - creates "Update connection" description

3. **Name Resolution**: Uses reverse mappings to get original display names

4. **Parameter Diff**: For same-type substitutions, includes parameter diff

=== Usage ===

This function is called internally by `calculate_graph_edit_distance()`. It's not typically called directly.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/n8n-io/n8n n8n]
* '''File:''' packages/@n8n/ai-workflow-builder.ee/evaluations/programmatic/python/src/similarity.py
* '''Lines:''' L223-386

=== Signature ===
<syntaxhighlight lang="python">
def _extract_operations_from_path(
    node_edit_path: List[tuple],
    edge_edit_path: List[tuple],
    g1: nx.DiGraph,
    g2: nx.DiGraph,
    config: WorkflowComparisonConfig,
    g1_name_mapping: Dict[str, str],
    g2_name_mapping: Dict[str, str],
) -> List[Dict[str, Any]]:
    """
    Extract edit operations from NetworkX's edit path.

    Args:
        node_edit_path: List of node edit tuples (u, v) where:
            - (u, v): nodes u in g1 and v in g2 are matched/substituted
            - (u, None): node u in g1 is deleted
            - (None, v): node v in g2 is inserted
        edge_edit_path: List of edge edit tuples ((u1, v1), (u2, v2))
        g1, g2: Relabeled graphs
        config: Configuration
        g1_name_mapping, g2_name_mapping: Mappings to original names

    Returns:
        List of edit operations with descriptions and costs
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# Internal function - not typically imported directly
from src.similarity import _extract_operations_from_path
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| node_edit_path || List[tuple] || Node edits from NetworkX
|-
| edge_edit_path || List[tuple] || Edge edits from NetworkX
|-
| g1 || nx.DiGraph || Relabeled generated graph
|-
| g2 || nx.DiGraph || Relabeled ground truth graph
|-
| config || WorkflowComparisonConfig || Cost configuration
|-
| g1_name_mapping || Dict[str, str] || Maps g1 structural IDs → original names
|-
| g2_name_mapping || Dict[str, str] || Maps g2 structural IDs → original names
|}

=== Output Operation Structure ===
{| class="wikitable"
|-
! Field !! Type !! When Present !! Description
|-
| type || str || Always || Operation type identifier
|-
| description || str || Always || Human-readable description
|-
| cost || float || Always || Operation cost from config
|-
| priority || str || Always || "critical", "major", or "minor"
|-
| node_name || str || Node ops || Original display name
|-
| parameter_diff || Dict || Same-type substitution || Added/removed/changed params
|}

== Usage Examples ==

=== Node Operation Processing ===
<syntaxhighlight lang="python">
# Node insertion (None, v)
# v = "node_0" in g2, maps to "HTTP Request"
{
    "type": "node_insert",
    "description": "Add missing node 'HTTP Request' (type: n8n-nodes-base.httpRequest)",
    "cost": 10.0,
    "priority": "major",
    "node_name": "HTTP Request"
}

# Node deletion (u, None)
# u = "trigger_0" in g1, maps to "Gmail Trigger"
{
    "type": "node_delete",
    "description": "Remove node 'Gmail Trigger' (type: n8n-nodes-base.gmailTrigger)",
    "cost": 10.0,
    "priority": "critical",  # Trigger deletion is critical
    "node_name": "Gmail Trigger"
}

# Node substitution - type change (u, v)
{
    "type": "node_substitute",
    "description": "Change node 'Webhook' from type 'n8n-nodes-base.webhook' to 'n8n-nodes-base.manualTrigger'",
    "cost": 50.0,  # Trigger mismatch
    "priority": "critical",
    "node_name": "Webhook"
}

# Node substitution - parameter update (u, v)
{
    "type": "node_substitute",
    "description": "Update parameters of node 'HTTP Request' (type: n8n-nodes-base.httpRequest)",
    "cost": 1.5,
    "priority": "minor",
    "node_name": "HTTP Request",
    "parameter_diff": {
        "changed": {
            "url": {"from": "http://old.com", "to": "https://new.com"}
        },
        "added": {"timeout": 30}
    }
}
</syntaxhighlight>

=== Edge Operation Processing ===
<syntaxhighlight lang="python">
# Edge insertion (None, e2)
{
    "type": "edge_insert",
    "description": "Add missing connection from 'Start' to 'HTTP Request'",
    "cost": 5.0,
    "priority": "major"
}

# Edge deletion (e1, None)
{
    "type": "edge_delete",
    "description": "Remove connection from 'Process' to 'End'",
    "cost": 5.0,
    "priority": "minor"
}
</syntaxhighlight>

=== Name Resolution Helper ===
<syntaxhighlight lang="python">
def get_display_name(node_id, mapping, graph):
    """Get original name from structural ID."""
    if mapping and node_id in mapping:
        return mapping[node_id]  # "trigger_0" → "Gmail Trigger"
    # Fallback to stored attribute
    return graph.nodes[node_id].get("_original_name", node_id)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:n8n-io_n8n_Edit_Operation_Extraction]]

=== Requires Environment ===
* [[requires_env::Environment:n8n-io_n8n_Workflow_Comparison_Env]]
