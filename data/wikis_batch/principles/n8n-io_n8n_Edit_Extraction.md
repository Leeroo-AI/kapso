{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|Graph Edit Distance|https://en.wikipedia.org/wiki/Graph_edit_distance]]
* [[source::Doc|NetworkX Edit Paths|https://networkx.org/documentation/stable/reference/algorithms/similarity.html]]
|-
! Domains
| [[domain::Graph_Algorithms]], [[domain::Workflow_Analysis]], [[domain::Data_Transformation]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

Edit Extraction is the principle of converting raw graph edit distance paths into human-readable operation descriptions that explain how one workflow differs from another.

=== Description ===

The edit extraction principle bridges the gap between algorithmic output and human understanding. While GED algorithms produce edit paths as sequences of mathematical operations on graph structures, users need clear, actionable descriptions of workflow differences.

The extraction process involves:

1. **Path Parsing**: Interpreting NetworkX edit path tuples representing operations

2. **Operation Classification**: Identifying the type of each operation:
   - Node insertion/deletion/substitution
   - Edge insertion/deletion/substitution
   - Parameter modifications

3. **Context Enrichment**: Adding descriptive information:
   - Node types and names
   - Parameter changes with before/after values
   - Connection details

4. **Cost Attribution**: Associating each operation with its configured cost

5. **Priority Ordering**: Sorting operations by importance (e.g., structural changes before parameter tweaks)

The result is a structured list of edits that can be displayed to users or processed by downstream systems.

=== Usage ===

Apply this principle when:
* Generating workflow diff reports for human review
* Creating automated workflow migration tools
* Building workflow version control systems
* Explaining similarity scores with concrete differences
* Generating change logs between workflow versions

== Theoretical Basis ==

=== Edit Path Structure ===

NetworkX returns edit paths as sequences of tuples:

```python
edit_path = [
    (node1_id, node2_id),  # Node mapping or operation
    ...
]

Special cases:
  (None, v):  Insert node v
  (u, None):  Delete node u
  (u, v):     Substitute/match node u with v
```

=== Operation Extraction Algorithm ===

```python
def extract_operations(edit_path, G1, G2):
    operations = []

    for (u, v) in edit_path:
        if u is None:
            # Node insertion
            op = {
                'type': 'insert_node',
                'node': v,
                'node_type': G2.nodes[v]['type'],
                'cost': config.node_insertion_cost
            }
        elif v is None:
            # Node deletion
            op = {
                'type': 'delete_node',
                'node': u,
                'node_type': G1.nodes[u]['type'],
                'cost': config.node_deletion_cost
            }
        else:
            # Node substitution or match
            if nodes_differ(G1.nodes[u], G2.nodes[v]):
                op = create_substitution(u, v, G1, G2)
            else:
                continue  # Matching node, no operation

        operations.append(op)

    return operations
```

=== Parameter Change Detection ===

For substitution operations, compare node attributes:

```python
def detect_parameter_changes(node1, node2):
    changes = []

    params1 = node1.get('parameters', {})
    params2 = node2.get('parameters', {})

    all_keys = set(params1.keys()) | set(params2.keys())

    for key in all_keys:
        if key in ignore_list:
            continue

        val1 = params1.get(key)
        val2 = params2.get(key)

        if val1 != val2:
            changes.append({
                'parameter': key,
                'old_value': val1,
                'new_value': val2
            })

    return changes
```

=== Operation Priority ===

Order operations by impact:

1. **High Priority**: Structural changes (node/edge insertion/deletion)
2. **Medium Priority**: Type changes (node substitutions)
3. **Low Priority**: Parameter modifications

=== Output Format ===

Structured operation representation:

```json
{
  "operation": "substitute_node",
  "source_node": "node_0",
  "target_node": "node_0",
  "changes": [
    {
      "type": "parameter_change",
      "parameter": "url",
      "old_value": "https://api.example.com/v1",
      "new_value": "https://api.example.com/v2"
    }
  ],
  "cost": 0.1
}
```

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:n8n-io_n8n_extract_operations_from_path]]

=== Related Principles ===
* [[related::Principle:n8n-io_n8n_GED_Calculation]]
* [[related::Principle:n8n-io_n8n_Result_Formatting]]
* [[related::Principle:n8n-io_n8n_Configuration_Loading]]
