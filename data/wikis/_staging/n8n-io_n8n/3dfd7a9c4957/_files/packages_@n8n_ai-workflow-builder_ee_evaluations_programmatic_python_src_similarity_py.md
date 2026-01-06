# File: `packages/@n8n/ai-workflow-builder.ee/evaluations/programmatic/python/src/similarity.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 501 |
| Functions | `calculate_graph_edit_distance` |
| Imports | networkx, src, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Calculate workflow graph edit distance

**Mechanism:** Uses NetworkX's graph edit distance algorithms with custom cost functions. calculate_graph_edit_distance() computes similarity scores between workflow graphs, _calculate_basic_edit_cost() provides fallback calculations, _extract_operations_from_path() details edit operations, and _determine_priority() assigns operation priorities.

**Significance:** Provides sophisticated workflow similarity scoring mechanism. Core algorithm for evaluating how well AI-generated workflows match reference workflows.
