# Implementation: build_workflow_graph

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
* [[source::Doc|NetworkX|https://networkx.org/documentation/stable/]]
|-
! Domains
| [[domain::Graph_Theory]], [[domain::Workflow_Evaluation]]
|-
! Last Updated
| [[last_updated::2024-12-18 12:00 GMT]]
|}

== Overview ==

Concrete function for converting n8n workflow JSON into a NetworkX directed graph with filtered attributes.

=== Description ===

`build_workflow_graph()` creates a directed graph from workflow JSON:

1. **Node Processing**: Iterates through `workflow["nodes"]`, filtering by configuration
2. **Attribute Extraction**: Extracts type, typeVersion, parameters, and trigger status
3. **Parameter Filtering**: Recursively filters parameters based on ignore rules
4. **Edge Processing**: Parses nested `connections` structure into graph edges
5. **Edge Attributes**: Includes connection_type, indices, and node type info

The function respects configuration for:
- Ignored node types (e.g., sticky notes)
- Ignored parameters (e.g., position, id)
- Ignored connection types

=== Usage ===

Call this function with parsed workflow JSON and optional configuration to get a graph suitable for GED calculation.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/n8n-io/n8n n8n]
* '''File:''' packages/@n8n/ai-workflow-builder.ee/evaluations/programmatic/python/src/graph_builder.py
* '''Lines:''' L10-90

=== Signature ===
<syntaxhighlight lang="python">
def build_workflow_graph(
    workflow: Dict[str, Any],
    config: Optional[WorkflowComparisonConfig] = None
) -> nx.DiGraph:
    """
    Convert n8n workflow to NetworkX directed graph.

    Args:
        workflow: n8n workflow JSON (with 'nodes' and 'connections')
        config: Optional configuration for filtering

    Returns:
        NetworkX DiGraph with nodes and edges
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from src.graph_builder import build_workflow_graph
import networkx as nx
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| workflow || Dict[str, Any] || Yes || Parsed workflow JSON with "nodes" and "connections"
|-
| config || Optional[WorkflowComparisonConfig] || No || Filtering and comparison configuration
|}

=== Outputs ===
{| class="wikitable"
|-
! Return Type !! Description
|-
| nx.DiGraph || Directed graph with workflow structure
|}

=== Node Attributes ===
{| class="wikitable"
|-
! Attribute !! Type !! Description
|-
| type || str || n8n node type (e.g., "n8n-nodes-base.httpRequest")
|-
| type_version || int || Node type version
|-
| parameters || Dict || Filtered parameter dictionary
|-
| is_trigger || bool || True if node is a trigger
|}

=== Edge Attributes ===
{| class="wikitable"
|-
! Attribute !! Type !! Description
|-
| connection_type || str || Connection type (usually "main")
|-
| source_index || int || Output index on source node
|-
| target_index || int || Input index on target node
|-
| source_node_type || str || Type of source node
|-
| target_node_type || str || Type of target node
|}

== Usage Examples ==

=== Basic Graph Building ===
<syntaxhighlight lang="python">
from src.graph_builder import build_workflow_graph
from src.config_loader import load_config

# Load workflow
workflow = {
    "nodes": [
        {"name": "Start", "type": "n8n-nodes-base.manualTrigger"},
        {"name": "HTTP", "type": "n8n-nodes-base.httpRequest",
         "parameters": {"url": "https://api.example.com"}}
    ],
    "connections": {
        "Start": {"main": [[{"node": "HTTP", "type": "main", "index": 0}]]}
    }
}

# Build graph with default config
config = load_config()
graph = build_workflow_graph(workflow, config)

print(f"Nodes: {list(graph.nodes())}")  # ['Start', 'HTTP']
print(f"Edges: {list(graph.edges())}")  # [('Start', 'HTTP')]
</syntaxhighlight>

=== Accessing Node Attributes ===
<syntaxhighlight lang="python">
graph = build_workflow_graph(workflow, config)

# Get node data
for node_name, data in graph.nodes(data=True):
    print(f"{node_name}:")
    print(f"  type: {data['type']}")
    print(f"  is_trigger: {data['is_trigger']}")
    print(f"  parameters: {data['parameters']}")
</syntaxhighlight>

=== Trigger Detection ===
<syntaxhighlight lang="python">
# _is_trigger_node() detects triggers by:
# 1. "trigger" in node type or name
# 2. Known trigger types: webhook, cron, manualtrigger, etc.

triggers = [
    node for node, data in graph.nodes(data=True)
    if data.get('is_trigger', False)
]
print(f"Trigger nodes: {triggers}")
</syntaxhighlight>

=== With Configuration Filtering ===
<syntaxhighlight lang="python">
from src.config_loader import WorkflowComparisonConfig

config = WorkflowComparisonConfig()
config.ignored_node_types = {"n8n-nodes-base.stickyNote"}
config.ignored_global_parameters = {"position", "id"}

# Build graph with filtering
graph = build_workflow_graph(workflow, config)
# Sticky notes excluded, position/id parameters filtered out
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:n8n-io_n8n_Graph_Construction]]

=== Requires Environment ===
* [[requires_env::Environment:n8n-io_n8n_Workflow_Comparison_Env]]
