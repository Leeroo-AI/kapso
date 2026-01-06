# Principle: Graph Construction

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

Principle for converting n8n workflow JSON structures into NetworkX directed graphs suitable for graph-based similarity analysis.

=== Description ===

Graph Construction transforms workflow JSON into a graph representation:

1. **Node Creation**: Each workflow node becomes a graph node with attributes (type, parameters, is_trigger)
2. **Edge Creation**: Each connection becomes a directed edge with connection metadata
3. **Filtering**: Configuration-based exclusion of ignored node types and parameters
4. **Attribute Extraction**: Trigger detection, parameter filtering, type extraction

The graph structure enables:
- Graph Edit Distance calculation
- Structural comparison independent of node names
- Parameter-level difference detection
- Topological analysis of workflow structure

=== Usage ===

Apply this principle when:
- Converting hierarchical structures to graphs
- Implementing graph-based comparison algorithms
- Building workflow analysis tools
- Creating visualization of workflow structures

== Theoretical Basis ==

Graph construction follows a **Structure Mapping** pattern:

<syntaxhighlight lang="python">
# Pseudo-code for graph construction

def build_workflow_graph(workflow: Dict, config: Config) -> nx.DiGraph:
    G = nx.DiGraph()

    # 1. Add nodes from workflow.nodes
    for node in workflow.get("nodes", []):
        if config.should_ignore_node(node):
            continue

        G.add_node(
            node["name"],
            type=node.get("type", ""),
            parameters=filter_parameters(node.get("parameters", {})),
            is_trigger=is_trigger_node(node),
        )

    # 2. Add edges from workflow.connections
    for source, conn_types in workflow.get("connections", {}).items():
        for conn_type, outputs in conn_types.items():
            for output_idx, targets in enumerate(outputs):
                for target in (targets or []):
                    if source in G and target["node"] in G:
                        G.add_edge(
                            source,
                            target["node"],
                            connection_type=target.get("type", "main"),
                        )

    return G
</syntaxhighlight>

Node attributes:
- **type**: n8n node type (e.g., "n8n-nodes-base.httpRequest")
- **parameters**: Filtered parameter dictionary
- **is_trigger**: Boolean indicating trigger nodes

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:n8n-io_n8n_build_workflow_graph]]
