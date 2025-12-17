{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|Graph Theory Fundamentals|https://en.wikipedia.org/wiki/Graph_theory]]
* [[source::Doc|NetworkX Documentation|https://networkx.org]]
* [[source::Doc|n8n Workflow Structure|https://docs.n8n.io]]
|-
! Domains
| [[domain::Graph_Algorithms]], [[domain::Workflow_Analysis]], [[domain::Data_Structures]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

Graph Construction is the principle of transforming n8n workflow JSON definitions into NetworkX directed graph representations where nodes represent workflow steps and edges represent data flow connections.

=== Description ===

The graph construction principle establishes a canonical mapping between n8n's workflow representation and graph theory data structures. This transformation enables the application of graph algorithms to workflow analysis problems.

The conversion process:

1. **Node Mapping**: Each workflow node (trigger, action, conditional) becomes a graph vertex with attributes:
   - Node type (e.g., "n8n-nodes-base.httpRequest")
   - Parameters (configuration settings)
   - Metadata (name, position, enabled status)

2. **Edge Mapping**: Each workflow connection becomes a directed graph edge representing:
   - Data flow from source to target node
   - Connection type (main data flow, error handling)
   - Output/input indices for multiple connections

3. **Attribute Preservation**: All relevant workflow properties are preserved as node/edge attributes for later comparison

This representation transforms workflow comparison into a graph isomorphism problem, enabling the use of well-studied algorithms like Graph Edit Distance.

=== Usage ===

Apply this principle when:
* Comparing workflow structures algorithmically
* Analyzing workflow topology and data flow
* Detecting workflow patterns or anti-patterns
* Visualizing workflow structure
* Computing workflow complexity metrics

== Theoretical Basis ==

=== Graph Representation ===

A workflow W is represented as a directed graph G = (V, E) where:

```
V = {v₁, v₂, ..., vₙ} (workflow nodes)
E = {(vᵢ, vⱼ) | connection from node i to node j}
```

Each vertex v ∈ V has attributes:
```
attr(v) = {type, parameters, metadata}
```

Each edge e ∈ E has attributes:
```
attr(e) = {source_output, target_input}
```

=== Directed Acyclic Graph Property ===

Valid n8n workflows form DAGs (Directed Acyclic Graphs):
* No cycles in data flow (preventing infinite loops)
* Clear topological ordering (execution sequence)
* Well-defined start nodes (triggers)

=== Transformation Algorithm ===

```python
def build_graph(workflow_json):
    G = nx.DiGraph()

    # Add nodes with attributes
    for node in workflow_json['nodes']:
        G.add_node(
            node['name'],
            node_type=node['type'],
            parameters=node['parameters']
        )

    # Add edges from connections
    for conn in workflow_json['connections']:
        source = conn['source']
        target = conn['target']
        G.add_edge(source, target)

    return G
```

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:n8n-io_n8n_build_workflow_graph]]

=== Related Principles ===
* [[related::Principle:n8n-io_n8n_Graph_Relabeling]]
* [[related::Principle:n8n-io_n8n_GED_Calculation]]
* [[related::Principle:n8n-io_n8n_Workflow_Loading]]
