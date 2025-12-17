{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|Graph Isomorphism|https://en.wikipedia.org/wiki/Graph_isomorphism]]
* [[source::Paper|Canonical Graph Labeling|https://dl.acm.org/doi/10.1145/321250.321254]]
|-
! Domains
| [[domain::Graph_Algorithms]], [[domain::Workflow_Analysis]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

Graph Relabeling is the principle of assigning positional identifiers to workflow nodes based on their structural role rather than user-defined names, enabling structural comparison independent of arbitrary naming choices.

=== Description ===

The graph relabeling principle addresses a fundamental challenge in workflow comparison: two workflows may be structurally identical but use different node names. By relabeling nodes according to their structural position, we can compare workflows based on topology and function rather than superficial naming differences.

The relabeling strategy:

1. **Trigger Identification**: Identify all trigger nodes (workflow entry points) and label them as `trigger_0`, `trigger_1`, etc.

2. **Topological Ordering**: Process remaining nodes in topological order (respecting data flow dependencies)

3. **Positional Labeling**: Assign sequential identifiers like `node_0`, `node_1`, etc. based on position in the execution graph

4. **Type Preservation**: Maintain node type and parameter information under the new labels

This transformation creates a canonical representation where node IDs reflect structural position, making it possible to detect equivalent workflows even when users have chosen different naming conventions.

=== Usage ===

Apply this principle when:
* Comparing workflows that may use different naming conventions
* Detecting duplicate or similar workflows across teams
* Building workflow search systems based on structural similarity
* Normalizing workflows for pattern detection
* Creating workflow templates from specific instances

== Theoretical Basis ==

=== Canonical Labeling Problem ===

Given a graph G = (V, E), find a labeling function f: V → L such that:
* f is bijective (one-to-one)
* Isomorphic graphs receive identical labels
* Labels reflect structural properties

=== Topological Sort for DAGs ===

For a DAG, topological ordering provides a natural canonical labeling:

```
topological_sort(G):
    L = empty list
    S = set of nodes with no incoming edges
    while S is not empty:
        remove node n from S
        add n to L
        for each node m with edge (n,m):
            remove edge (n,m)
            if m has no incoming edges:
                add m to S
    return L
```

=== Relabeling Algorithm ===

```python
def relabel_by_structure(G):
    # Identify triggers (no incoming edges)
    triggers = [n for n in G.nodes() if G.in_degree(n) == 0]

    # Label triggers
    label_map = {}
    for i, trigger in enumerate(sorted(triggers)):
        label_map[trigger] = f"trigger_{i}"

    # Topological sort for remaining nodes
    remaining = [n for n in G.nodes() if n not in triggers]
    topo_order = nx.topological_sort(G.subgraph(remaining))

    # Label remaining nodes
    for i, node in enumerate(topo_order):
        label_map[node] = f"node_{i}"

    # Relabel graph
    return nx.relabel_nodes(G, label_map)
```

=== Structural Equivalence ===

Two workflows W₁ and W₂ are structurally equivalent if:
```
relabel(G₁) ≅ relabel(G₂)
```
where ≅ denotes graph isomorphism.

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:n8n-io_n8n_relabel_graph_by_structure]]

=== Related Principles ===
* [[related::Principle:n8n-io_n8n_Graph_Construction]]
* [[related::Principle:n8n-io_n8n_GED_Calculation]]
