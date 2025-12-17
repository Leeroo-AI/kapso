# File: `packages/@n8n/ai-workflow-builder.ee/evaluations/programmatic/python/src/similarity.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 501 |
| Functions | `calculate_graph_edit_distance` |
| Imports | networkx, src, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Calculate workflow similarity using graph edit distance

**Mechanism:** The calculate_graph_edit_distance() function uses NetworkX's optimize_edit_paths() with custom cost functions and edge matching. Key algorithm steps: (1) relabel graphs by structure rather than names using _relabel_graph_by_structure() which sorts nodes by type/degree and creates consistent IDs like trigger_0, node_1, (2) apply GED algorithm with node cost functions and edge_match function for connection type equivalence, (3) extract edit operations from the optimal path using _extract_operations_from_path(), (4) calculate similarity score as 1 - (edit_cost / max_possible_cost). Determines priority levels (critical/major/minor) based on cost thresholds and operation types. Includes fallback to basic edit cost calculation if GED fails. Preserves original names in node attributes and uses normalized name hashing to prevent inappropriate node swapping.

**Significance:** The core algorithm that quantifies workflow similarity. Graph edit distance is mathematically rigorous and captures both structural differences (missing/extra nodes and connections) and content differences (parameter variations). The structural relabeling ensures nodes are matched by their role in the workflow rather than arbitrary names, which is critical for comparing AI-generated workflows that may use different naming conventions. This produces actionable insights about what needs to be changed to improve generated workflows.
