{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|Graph Edit Distance|https://en.wikipedia.org/wiki/Graph_edit_distance]]
* [[source::Paper|A* Algorithm for GED|https://doi.org/10.1016/j.patcog.2008.08.026]]
* [[source::Doc|NetworkX optimize_edit_paths|https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.similarity.optimize_edit_paths.html]]
|-
! Domains
| [[domain::Graph_Algorithms]], [[domain::Similarity_Metrics]], [[domain::Workflow_Analysis]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

Graph Edit Distance (GED) Calculation is the principle of measuring workflow similarity by computing the minimum cost sequence of edit operations needed to transform one workflow graph into another.

=== Description ===

The GED calculation principle provides a quantitative measure of how different two workflows are by determining the least expensive set of changes needed to make them identical. This approach treats workflow comparison as an optimization problem.

Edit operations include:

1. **Node Operations**:
   - **Insertion**: Add a new node to the workflow
   - **Deletion**: Remove an existing node
   - **Substitution**: Replace one node with another (type or parameter change)

2. **Edge Operations**:
   - **Insertion**: Add a new connection between nodes
   - **Deletion**: Remove a connection
   - **Substitution**: Modify connection properties

Each operation has an associated cost based on the configuration. The GED is the sum of costs for the cheapest transformation sequence.

The problem is NP-hard in general, but practical for typical workflow sizes (10-100 nodes). NetworkX's `optimize_edit_paths` uses A* search with heuristics to find optimal solutions efficiently.

=== Usage ===

Apply this principle when:
* Computing precise similarity scores between workflows
* Identifying what changes were made between workflow versions
* Finding the most similar workflow in a repository
* Generating automated workflow migration suggestions
* Validating workflow refactoring operations

== Theoretical Basis ==

=== Formal Definition ===

Given two graphs G₁ = (V₁, E₁) and G₂ = (V₂, E₂), the Graph Edit Distance is:

```
GED(G₁, G₂) = min Σ cost(op)
              p∈P
```

where P is the set of all edit paths transforming G₁ into G₂, and cost(op) is the cost function for each edit operation.

=== Edit Operations ===

The six fundamental operations:

```
Node Operations:
  insert_node(v):  V → V ∪ {v}
  delete_node(v):  V → V \ {v}
  substitute_node(u, v):  Replace u with v

Edge Operations:
  insert_edge(u, v):  E → E ∪ {(u,v)}
  delete_edge(u, v):  E → E \ {(u,v)}
  substitute_edge((u,v), (x,y)):  Replace edge
```

=== Cost Function ===

A weighted cost function prioritizes different types of changes:

```python
cost(operation) = {
    w_ni  if operation = insert_node
    w_nd  if operation = delete_node
    w_ns  if operation = substitute_node
    w_ei  if operation = insert_edge
    w_ed  if operation = delete_edge
    w_es  if operation = substitute_edge
}
```

=== A* Search Algorithm ===

NetworkX uses A* to find optimal edit paths:

```
f(n) = g(n) + h(n)

where:
  g(n) = actual cost from start to node n
  h(n) = heuristic estimate of cost from n to goal
  f(n) = estimated total cost
```

The heuristic ensures admissibility (never overestimates), guaranteeing optimal solutions.

=== Node/Edge Matching ===

Custom matching functions determine substitution costs:

```python
def node_match(n1, n2):
    # Compare node types and parameters
    if n1['type'] != n2['type']:
        return False
    return parameter_similarity(n1, n2) > threshold

def edge_match(e1, e2):
    # Compare edge attributes
    return e1['connection_type'] == e2['connection_type']
```

=== Complexity ===

* **Theoretical**: O(n!) for n nodes (exponential)
* **Practical**: A* with good heuristics handles workflows up to ~100 nodes efficiently
* **Trade-offs**: Exact computation vs. approximate algorithms for large graphs

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:n8n-io_n8n_calculate_graph_edit_distance]]

=== Related Principles ===
* [[related::Principle:n8n-io_n8n_Graph_Construction]]
* [[related::Principle:n8n-io_n8n_Graph_Relabeling]]
* [[related::Principle:n8n-io_n8n_Configuration_Loading]]
* [[related::Principle:n8n-io_n8n_Edit_Extraction]]
* [[related::Principle:n8n-io_n8n_Similarity_Calculation]]
