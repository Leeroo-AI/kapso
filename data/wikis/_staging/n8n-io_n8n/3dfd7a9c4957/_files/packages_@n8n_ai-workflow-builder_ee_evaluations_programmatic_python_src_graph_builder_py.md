# File: `packages/@n8n/ai-workflow-builder.ee/evaluations/programmatic/python/src/graph_builder.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 222 |
| Functions | `build_workflow_graph`, `get_node_data`, `get_edge_data`, `graph_stats` |
| Imports | networkx, src, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Convert workflows to graph representation

**Mechanism:** Transforms n8n workflow JSON into NetworkX directed graphs. build_workflow_graph() creates graph nodes with filtered parameters, _filter_parameters() removes irrelevant fields, _is_trigger_node() detects trigger nodes, and graph_stats() generates workflow statistics.

**Significance:** Enables graph-based workflow comparison and analysis. Foundation for the similarity calculation pipeline - workflows must be converted to graphs before edit distance can be computed.
