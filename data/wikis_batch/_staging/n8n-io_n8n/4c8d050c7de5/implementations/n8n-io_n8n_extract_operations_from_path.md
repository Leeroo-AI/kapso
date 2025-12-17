{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
|-
! Domains
| [[domain::Graph_Algorithms]], [[domain::Diff_Generation]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

Concrete tool for extracting and formatting edit operations from NetworkX's graph edit distance path, provided by the n8n workflow comparison system.

=== Description ===

The `_extract_operations_from_path` function translates the raw edit path returned by NetworkX's graph edit distance algorithm into a human-readable list of operations. It:
* Converts node edit operations (insertions, deletions, substitutions) into descriptive entries
* Converts edge edit operations into descriptive entries
* Maps structural IDs back to original node names
* Enriches operations with node types and cost information
* Creates actionable descriptions for each difference

This function bridges the gap between the algorithmic output and user-facing reporting.

=== Usage ===

Use this implementation when you need to:
* Generate human-readable diff reports for workflow comparisons
* Extract specific edit operations from graph comparison results
* Translate structural IDs back to original workflow node names
* Create actionable feedback for workflow authors

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/n8n-io/n8n n8n]
* '''File:''' packages/@n8n/ai-workflow-builder.ee/evaluations/programmatic/python/src/similarity.py
* '''Lines:''' L223-386

=== Signature ===
<syntaxhighlight lang="python">
def _extract_operations_from_path(
    node_edit_path: List[Tuple],
    edge_edit_path: List[Tuple],
    g1: nx.DiGraph,
    g2: nx.DiGraph,
    config: WorkflowComparisonConfig,
    g1_mapping: Dict[str, str],
    g2_mapping: Dict[str, str]
) -> List[Dict[str, Any]]:
    """Extract edit operations from NetworkX's edit path.

    Args:
        node_edit_path: List of node edit operations from NetworkX
        edge_edit_path: List of edge edit operations from NetworkX
        g1: First relabeled graph
        g2: Second relabeled graph
        config: Configuration with cost functions
        g1_mapping: Mapping from structural IDs to original names for g1
        g2_mapping: Mapping from structural IDs to original names for g2

    Returns:
        List of operation dictionaries with type, description, cost, and details
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from src.similarity import _extract_operations_from_path
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| node_edit_path || List[Tuple] || Yes || Node edit operations as (u, v) tuples from NetworkX
|-
| edge_edit_path || List[Tuple] || Yes || Edge edit operations as ((u1,v1), (u2,v2)) tuples from NetworkX
|-
| g1 || nx.DiGraph || Yes || First graph (relabeled with structural IDs)
|-
| g2 || nx.DiGraph || Yes || Second graph (relabeled with structural IDs)
|-
| config || WorkflowComparisonConfig || Yes || Configuration with cost functions
|-
| g1_mapping || Dict[str, str] || Yes || Structural ID to original name mapping for g1
|-
| g2_mapping || Dict[str, str] || Yes || Structural ID to original name mapping for g2
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| operations || List[Dict[str, Any]] || List of operation dictionaries
|}

=== Operation Dictionary Structure ===
Each operation dictionary contains:
{| class="wikitable"
|-
! Field !! Type !! Description
|-
| type || str || Operation type: "node_insert", "node_delete", "node_substitute", "edge_insert", "edge_delete"
|-
| description || str || Human-readable description (e.g., "Add missing node 'HTTP Request'")
|-
| cost || float || Cost contribution of this operation
|-
| details || Dict || Additional context specific to operation type
|}

=== Operation Types and Details ===

'''node_insert:'''
<syntaxhighlight lang="python">
{
    "type": "node_insert",
    "description": "Add missing node 'NodeName' (type: node-type)",
    "cost": 10.0,
    "details": {
        "node": "NodeName",
        "node_type": "n8n-nodes-base.httpRequest",
        "in_graph": 2
    }
}
</syntaxhighlight>

'''node_delete:'''
<syntaxhighlight lang="python">
{
    "type": "node_delete",
    "description": "Remove extra node 'NodeName' (type: node-type)",
    "cost": 10.0,
    "details": {
        "node": "NodeName",
        "node_type": "n8n-nodes-base.set",
        "in_graph": 1
    }
}
</syntaxhighlight>

'''node_substitute:'''
<syntaxhighlight lang="python">
{
    "type": "node_substitute",
    "description": "Replace node 'Node1' (type1) with 'Node2' (type2)",
    "cost": 5.0,
    "details": {
        "from_node": "Node1",
        "to_node": "Node2",
        "from_type": "n8n-nodes-base.httpRequest",
        "to_type": "n8n-nodes-base.webhook"
    }
}
</syntaxhighlight>

'''edge_insert:'''
<syntaxhighlight lang="python">
{
    "type": "edge_insert",
    "description": "Add missing connection: Node1 -> Node2",
    "cost": 5.0,
    "details": {
        "from_node": "Node1",
        "to_node": "Node2",
        "connection_type": "main",
        "in_graph": 2
    }
}
</syntaxhighlight>

'''edge_delete:'''
<syntaxhighlight lang="python">
{
    "type": "edge_delete",
    "description": "Remove extra connection: Node1 -> Node2",
    "cost": 5.0,
    "details": {
        "from_node": "Node1",
        "to_node": "Node2",
        "connection_type": "main",
        "in_graph": 1
    }
}
</syntaxhighlight>

== Usage Examples ==

=== Basic Operation Extraction ===
<syntaxhighlight lang="python">
import networkx as nx
from src.similarity import _relabel_graph_by_structure, _extract_operations_from_path
from src.config_loader import load_config

# Prepare graphs
config = load_config("preset:balanced")
g1_relabeled, g1_mapping = _relabel_graph_by_structure(graph1)
g2_relabeled, g2_mapping = _relabel_graph_by_structure(graph2)

# Get edit path from NetworkX
edit_path_gen = nx.optimize_edit_paths(
    g1_relabeled,
    g2_relabeled,
    # ... cost functions ...
)

for node_edit_path, edge_edit_path, cost in edit_path_gen:
    # Extract operations
    operations = _extract_operations_from_path(
        node_edit_path,
        edge_edit_path,
        g1_relabeled,
        g2_relabeled,
        config,
        g1_mapping,
        g2_mapping
    )

    # Print operations
    for op in operations:
        print(f"[{op['type']}] {op['description']} (cost: {op['cost']})")

    break  # Only need first path
</syntaxhighlight>

=== Filtering Operations by Type ===
<syntaxhighlight lang="python">
# Extract operations
operations = _extract_operations_from_path(
    node_edit_path, edge_edit_path, g1, g2, config, g1_map, g2_map
)

# Filter by operation type
node_insertions = [op for op in operations if op['type'] == 'node_insert']
node_deletions = [op for op in operations if op['type'] == 'node_delete']
node_substitutions = [op for op in operations if op['type'] == 'node_substitute']
edge_changes = [op for op in operations if op['type'].startswith('edge_')]

print(f"Missing nodes (need to add): {len(node_insertions)}")
for op in node_insertions:
    print(f"  - {op['details']['node']} ({op['details']['node_type']})")

print(f"\nExtra nodes (need to remove): {len(node_deletions)}")
for op in node_deletions:
    print(f"  - {op['details']['node']} ({op['details']['node_type']})")

print(f"\nNode type mismatches: {len(node_substitutions)}")
for op in node_substitutions:
    print(f"  - {op['details']['from_node']}: {op['details']['from_type']} -> {op['details']['to_type']}")

print(f"\nEdge changes: {len(edge_changes)}")
</syntaxhighlight>

=== Calculating Total Cost by Category ===
<syntaxhighlight lang="python">
from collections import defaultdict

# Extract operations
operations = _extract_operations_from_path(
    node_edit_path, edge_edit_path, g1, g2, config, g1_map, g2_map
)

# Calculate costs by category
costs_by_type = defaultdict(float)
for op in operations:
    costs_by_type[op['type']] += op['cost']

total_cost = sum(costs_by_type.values())

print("Cost breakdown:")
print(f"  Node insertions:    {costs_by_type['node_insert']:.1f}")
print(f"  Node deletions:     {costs_by_type['node_delete']:.1f}")
print(f"  Node substitutions: {costs_by_type['node_substitute']:.1f}")
print(f"  Edge insertions:    {costs_by_type['edge_insert']:.1f}")
print(f"  Edge deletions:     {costs_by_type['edge_delete']:.1f}")
print(f"  Total:              {total_cost:.1f}")
</syntaxhighlight>

=== Generating User-Friendly Diff Report ===
<syntaxhighlight lang="python">
def generate_diff_report(operations):
    """Generate human-readable diff report."""
    report_lines = []
    report_lines.append("Workflow Comparison Report")
    report_lines.append("=" * 50)
    report_lines.append("")

    # Group operations by category
    categories = {
        "node_insert": "Missing Nodes (needs to be added)",
        "node_delete": "Extra Nodes (needs to be removed)",
        "node_substitute": "Node Type Mismatches",
        "edge_insert": "Missing Connections",
        "edge_delete": "Extra Connections",
    }

    for op_type, category_name in categories.items():
        ops = [op for op in operations if op['type'] == op_type]
        if not ops:
            continue

        report_lines.append(f"\n{category_name} ({len(ops)}):")
        report_lines.append("-" * 50)

        for op in ops:
            report_lines.append(f"  {op['description']}")
            report_lines.append(f"    Cost: {op['cost']}")

    return "\n".join(report_lines)

# Generate report
operations = _extract_operations_from_path(
    node_edit_path, edge_edit_path, g1, g2, config, g1_map, g2_map
)
report = generate_diff_report(operations)
print(report)
</syntaxhighlight>

=== Converting Operations to Action Items ===
<syntaxhighlight lang="python">
def operations_to_action_items(operations):
    """Convert operations to actionable checklist."""
    actions = []

    for op in operations:
        if op['type'] == 'node_insert':
            action = f"[ ] Add node: {op['details']['node']} (type: {op['details']['node_type']})"
        elif op['type'] == 'node_delete':
            action = f"[ ] Remove node: {op['details']['node']}"
        elif op['type'] == 'node_substitute':
            action = f"[ ] Change node '{op['details']['from_node']}' from {op['details']['from_type']} to {op['details']['to_type']}"
        elif op['type'] == 'edge_insert':
            action = f"[ ] Add connection: {op['details']['from_node']} -> {op['details']['to_node']}"
        elif op['type'] == 'edge_delete':
            action = f"[ ] Remove connection: {op['details']['from_node']} -> {op['details']['to_node']}"

        actions.append(action)

    return actions

# Generate action items
operations = _extract_operations_from_path(
    node_edit_path, edge_edit_path, g1, g2, config, g1_map, g2_map
)
actions = operations_to_action_items(operations)

print("Action Items to Match Reference Workflow:")
for action in actions:
    print(action)
</syntaxhighlight>

=== Exporting Operations to JSON ===
<syntaxhighlight lang="python">
import json

# Extract operations
operations = _extract_operations_from_path(
    node_edit_path, edge_edit_path, g1, g2, config, g1_map, g2_map
)

# Export to JSON for further processing
output = {
    "total_operations": len(operations),
    "operations": operations,
    "summary": {
        "node_inserts": sum(1 for op in operations if op['type'] == 'node_insert'),
        "node_deletes": sum(1 for op in operations if op['type'] == 'node_delete'),
        "node_substitutes": sum(1 for op in operations if op['type'] == 'node_substitute'),
        "edge_inserts": sum(1 for op in operations if op['type'] == 'edge_insert'),
        "edge_deletes": sum(1 for op in operations if op['type'] == 'edge_delete'),
    }
}

with open('operations.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"Exported {len(operations)} operations to operations.json")
</syntaxhighlight>

== Implementation Details ==

=== Edit Path Format ===
NetworkX returns edit paths as tuples:
* '''Node operations:''' `(u, v)` where:
  * `u is None, v is not None`: Node insertion (add v to g1)
  * `u is not None, v is None`: Node deletion (remove u from g1)
  * `u is not None, v is not None`: Node substitution (replace u with v)

* '''Edge operations:''' `((u1, v1), (u2, v2))` where:
  * First tuple is `None`: Edge insertion
  * Second tuple is `None`: Edge deletion
  * Both present: Edge substitution (treated as delete + insert)

=== Mapping Translation ===
The function uses `g1_mapping` and `g2_mapping` to translate structural IDs (trigger_0, node_1) back to original node names for user-friendly output.

=== Cost Calculation ===
Each operation's cost is calculated using the configuration's cost functions, ensuring consistency with the overall edit distance calculation.

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:n8n-io_n8n_Edit_Extraction]]

=== Used By ===
* [[used_by::Implementation:n8n-io_n8n_calculate_graph_edit_distance]]

=== Uses ===
* [[uses::Implementation:n8n-io_n8n_relabel_graph_by_structure]]

=== Requires Environment ===
* [[requires_env::Environment:n8n-io_n8n_Python_Workflow_Comparison]]
* [[requires_dependency::networkx]]

[[Category:Implementation]]
[[Category:n8n]]
[[Category:Graph_Algorithms]]
