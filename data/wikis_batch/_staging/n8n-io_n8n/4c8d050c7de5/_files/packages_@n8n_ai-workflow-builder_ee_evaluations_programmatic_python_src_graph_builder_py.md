# File: `packages/@n8n/ai-workflow-builder.ee/evaluations/programmatic/python/src/graph_builder.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 222 |
| Functions | `build_workflow_graph`, `get_node_data`, `get_edge_data`, `graph_stats` |
| Imports | networkx, src, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Convert n8n workflows to NetworkX graphs

**Mechanism:** The build_workflow_graph() function transforms n8n workflow JSON (with nodes and connections) into NetworkX DiGraph objects. Applies config-based filtering to ignore certain nodes, node types, and parameters. For each node, extracts type, typeVersion, filtered parameters, and trigger status (detected via _is_trigger_node() checking for 'trigger' in name/type or known trigger types like webhook, manualTrigger). For edges, extracts connection metadata including type, indices, and source/target node types. Helper functions provide graph statistics (node count, edge count, trigger count, node types, connectivity) and data access methods.

**Significance:** Bridge between n8n's workflow representation and graph theory algorithms. Enables sophisticated workflow comparison by modeling workflows as directed graphs where nodes represent workflow steps and edges represent data flow. The filtering capabilities ensure comparisons focus on semantically meaningful differences. Essential first step in the workflow comparison pipeline.
