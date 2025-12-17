{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
|-
! Domains
| [[domain::Graph_Algorithms]], [[domain::Workflow_Analysis]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

Concrete tool for converting n8n workflow JSON structures into NetworkX directed graphs, provided by the n8n workflow comparison system.

=== Description ===

The `build_workflow_graph` function transforms n8n workflow definitions (JSON format) into NetworkX directed graph representations. This conversion enables graph-theoretic analysis, including:
* Graph edit distance calculation
* Structural similarity comparison
* Topology analysis
* Node and edge relationship mapping

The function respects configuration settings to filter out ignored nodes and edges, and enriches graph elements with metadata for comparison purposes.

=== Usage ===

Use this implementation when you need to:
* Convert workflows from JSON to graph representation
* Prepare workflows for structural comparison
* Analyze workflow topology and dependencies
* Apply configuration-based filtering during graph construction

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
    """Convert n8n workflow to NetworkX directed graph.

    Args:
        workflow: n8n workflow dictionary containing nodes and connections
        config: Optional configuration for filtering and comparison settings

    Returns:
        NetworkX directed graph representing the workflow structure
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from src.graph_builder import build_workflow_graph
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| workflow || Dict[str, Any] || Yes || n8n workflow JSON structure containing 'nodes' and 'connections' keys
|-
| config || Optional[WorkflowComparisonConfig] || No || Configuration object for filtering nodes/edges and comparison settings
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| graph || nx.DiGraph || NetworkX directed graph with nodes and edges representing workflow structure
|}

=== Graph Node Attributes ===
{| class="wikitable"
|-
! Attribute !! Type !! Description
|-
| type || str || Node type (e.g., "n8n-nodes-base.httpRequest")
|-
| is_trigger || bool || Whether the node is a trigger node
|-
| position || tuple || (x, y) coordinates from workflow canvas
|-
| parameters || dict || Node configuration parameters
|}

=== Graph Edge Attributes ===
{| class="wikitable"
|-
! Attribute !! Type !! Description
|-
| connection_type || str || Type of connection (typically "main")
|-
| source_output || int || Output index from source node
|-
| target_input || int || Input index to target node
|}

== Usage Examples ==

=== Basic Graph Construction ===
<syntaxhighlight lang="python">
from src.compare_workflows import load_workflow
from src.graph_builder import build_workflow_graph

# Load workflow and build graph
workflow = load_workflow("workflows/my_workflow.json")
graph = build_workflow_graph(workflow)

# Inspect graph properties
print(f"Number of nodes: {graph.number_of_nodes()}")
print(f"Number of edges: {graph.number_of_edges()}")
print(f"Node types: {[graph.nodes[n]['type'] for n in graph.nodes()]}")
</syntaxhighlight>

=== Graph Construction with Configuration ===
<syntaxhighlight lang="python">
from src.config_loader import load_config
from src.compare_workflows import load_workflow
from src.graph_builder import build_workflow_graph

# Load configuration that ignores sticky notes
config = load_config("preset:strict")

# Build graph with filtering
workflow = load_workflow("workflows/annotated_workflow.json")
graph = build_workflow_graph(workflow, config)

# Sticky notes and other ignored nodes are excluded
print(f"Filtered nodes: {graph.number_of_nodes()}")
</syntaxhighlight>

=== Analyzing Graph Structure ===
<syntaxhighlight lang="python">
import networkx as nx

# Build graph
workflow = load_workflow("workflows/complex_workflow.json")
graph = build_workflow_graph(workflow)

# Find trigger nodes
triggers = [n for n, d in graph.nodes(data=True) if d.get('is_trigger', False)]
print(f"Trigger nodes: {triggers}")

# Find leaf nodes (no outgoing edges)
leaf_nodes = [n for n in graph.nodes() if graph.out_degree(n) == 0]
print(f"Leaf nodes: {leaf_nodes}")

# Calculate graph metrics
print(f"Is DAG: {nx.is_directed_acyclic_graph(graph)}")
print(f"Average degree: {sum(dict(graph.degree()).values()) / graph.number_of_nodes():.2f}")
</syntaxhighlight>

=== Building Graphs for Comparison ===
<syntaxhighlight lang="python">
from src.config_loader import load_config
from src.compare_workflows import load_workflow
from src.graph_builder import build_workflow_graph
from src.similarity import calculate_graph_edit_distance

# Load configuration
config = load_config("preset:balanced")

# Load workflows
workflow1 = load_workflow("workflows/version1.json")
workflow2 = load_workflow("workflows/version2.json")

# Build graphs with same configuration
graph1 = build_workflow_graph(workflow1, config)
graph2 = build_workflow_graph(workflow2, config)

# Compare graphs
result = calculate_graph_edit_distance(graph1, graph2, config)
print(f"Similarity: {result['similarity_score']:.2%}")
</syntaxhighlight>

=== Visualizing Workflow Graph ===
<syntaxhighlight lang="python">
import matplotlib.pyplot as plt
import networkx as nx

# Build graph
workflow = load_workflow("workflows/simple_workflow.json")
graph = build_workflow_graph(workflow)

# Extract positions from workflow canvas
pos = {node: graph.nodes[node].get('position', (0, 0)) for node in graph.nodes()}

# Draw graph
plt.figure(figsize=(12, 8))
nx.draw(graph, pos, with_labels=True, node_color='lightblue',
        node_size=2000, font_size=10, arrows=True)
plt.title("Workflow Graph Structure")
plt.axis('off')
plt.tight_layout()
plt.savefig("workflow_graph.png")
</syntaxhighlight>

== Implementation Details ==

=== Node Filtering ===
The function checks `config.should_ignore_node(node)` to determine if a node should be excluded based on:
* Node type (e.g., sticky notes, annotations)
* Custom filtering rules in configuration

=== Connection Processing ===
The function processes the n8n connections structure:
<syntaxhighlight lang="python">
# n8n connection format
"connections": {
  "sourceNode": {
    "main": [
      [{"node": "targetNode", "type": "main", "index": 0}]
    ]
  }
}
</syntaxhighlight>

And converts it to graph edges with appropriate attributes.

=== Trigger Node Detection ===
The function uses `_is_trigger_node(node)` to identify trigger nodes, which is important for:
* Workflow entry point identification
* Structural relabeling during comparison
* Topology analysis

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:n8n-io_n8n_Graph_Construction]]

=== Uses ===
* [[uses::Implementation:n8n-io_n8n_load_workflow]]
* [[uses::Implementation:n8n-io_n8n_load_config]]

=== Used By ===
* [[used_by::Implementation:n8n-io_n8n_calculate_graph_edit_distance]]
* [[used_by::Implementation:n8n-io_n8n_relabel_graph_by_structure]]

=== Requires Environment ===
* [[requires_env::Environment:n8n-io_n8n_Python_Workflow_Comparison]]
* [[requires_dependency::networkx]]

[[Category:Implementation]]
[[Category:n8n]]
[[Category:Graph_Algorithms]]
