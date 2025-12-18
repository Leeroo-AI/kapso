# Principle: Graph Relabeling

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

Principle for relabeling graph nodes from display names to structural identifiers to enable comparison by structure rather than naming.

=== Description ===

Graph Relabeling transforms node identifiers for fair comparison:

1. **Name Independence**: Replaces user-defined names with structural IDs
2. **Trigger Prioritization**: Trigger nodes labeled separately (`trigger_0`, `trigger_1`)
3. **Structural Sorting**: Nodes sorted by type and degree for consistent ordering
4. **Mapping Preservation**: Original names stored for error message display

Benefits:
- Workflows with different node names can be compared fairly
- AI-generated workflows don't get penalized for naming differences
- Structural similarity is the focus, not naming conventions
- Original names available for human-readable output

=== Usage ===

Apply this principle when:
- Comparing graphs where node identity is arbitrary
- Implementing structure-based graph matching
- Building systems where node names are cosmetic
- Creating comparison tools for user-generated content

== Theoretical Basis ==

Relabeling follows a **Canonical Labeling** approach:

<syntaxhighlight lang="python">
# Pseudo-code for graph relabeling

def relabel_graph_by_structure(graph: nx.DiGraph) -> tuple[nx.DiGraph, Dict]:
    nodes_with_data = list(graph.nodes(data=True))

    # 1. Sort by structural properties
    def sort_key(node_tuple):
        name, data = node_tuple
        return (
            data.get("type", ""),
            -graph.out_degree(name),  # Higher out-degree first
            -graph.in_degree(name),   # Higher in-degree first
            name,                      # Deterministic tiebreaker
        )

    # 2. Separate triggers and non-triggers
    triggers = sorted([n for n in nodes_with_data if n[1].get("is_trigger")], key=sort_key)
    non_triggers = sorted([n for n in nodes_with_data if not n[1].get("is_trigger")], key=sort_key)

    # 3. Assign structural labels
    mapping = {}
    reverse_mapping = {}

    for i, (name, _) in enumerate(triggers):
        new_label = f"trigger_{i}"
        mapping[name] = new_label
        reverse_mapping[new_label] = name

    for i, (name, _) in enumerate(non_triggers):
        new_label = f"node_{i}"
        mapping[name] = new_label
        reverse_mapping[new_label] = name

    # 4. Create relabeled graph
    relabeled = nx.relabel_nodes(graph, mapping, copy=True)

    return relabeled, reverse_mapping
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:n8n-io_n8n_relabel_graph_by_structure]]
