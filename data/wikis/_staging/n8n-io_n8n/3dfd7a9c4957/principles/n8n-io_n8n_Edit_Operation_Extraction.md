# Principle: Edit Operation Extraction

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

Principle for converting raw GED edit paths into human-readable operation descriptions with costs, priorities, and original node names.

=== Description ===

Edit Operation Extraction transforms NetworkX edit paths into actionable feedback:

1. **Node Path Processing**: Interprets (u, v) tuples as insert, delete, or substitute
2. **Edge Path Processing**: Interprets edge tuples as connection changes
3. **Name Resolution**: Maps structural IDs back to original node names
4. **Cost Assignment**: Calculates cost for each operation using config
5. **Priority Assignment**: Categorizes operations as critical/major/minor
6. **Description Generation**: Creates human-readable descriptions

The output provides:
- Clear descriptions for users ("Add missing node 'HTTP Request'")
- Cost information for debugging
- Priority for triage of issues
- Parameter diffs for same-type substitutions

=== Usage ===

Apply this principle when:
- Converting algorithm output to user feedback
- Building diff visualization tools
- Creating actionable error messages
- Implementing code review or comparison tools

== Theoretical Basis ==

Edit path interpretation:

<syntaxhighlight lang="python">
# Edit path tuples from NetworkX:

# Node operations: List[tuple[node_id, node_id]]
# - (u, v) where both non-None: substitute u with v
# - (u, None): delete u
# - (None, v): insert v

# Edge operations: List[tuple[edge, edge]]
# - (e1, e2) where both non-None: substitute e1 with e2
# - (e1, None): delete e1
# - (None, e2): insert e2

for u, v in node_edit_path:
    if u is None:
        # Node insertion (v in g2 is inserted)
        ops.append({"type": "node_insert", "description": f"Add '{name}'"})
    elif v is None:
        # Node deletion (u in g1 is deleted)
        ops.append({"type": "node_delete", "description": f"Remove '{name}'"})
    else:
        # Node substitution (u matched to v)
        if types_differ:
            ops.append({"type": "node_substitute", "description": f"Change type..."})
        else:
            ops.append({"type": "node_substitute", "description": f"Update params..."})
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:n8n-io_n8n_extract_operations_from_path]]
