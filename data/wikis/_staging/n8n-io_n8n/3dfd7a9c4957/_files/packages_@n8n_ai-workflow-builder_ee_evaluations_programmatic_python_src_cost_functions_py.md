# File: `packages/@n8n/ai-workflow-builder.ee/evaluations/programmatic/python/src/cost_functions.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 497 |
| Functions | `normalize_expression`, `node_substitution_cost`, `node_deletion_cost`, `node_insertion_cost`, `edge_substitution_cost`, `edge_deletion_cost`, `edge_insertion_cost`, `compare_parameters`, `... +5 more` |
| Imports | re, src, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Compute workflow edit operation costs

**Mechanism:** Provides sophisticated cost calculation functions for graph edit operations including node substitution/deletion/insertion and edge modifications. Uses normalize_expression() for consistent comparison, apply_comparison_rule() for custom rules, and calculate_semantic_similarity() for parameter matching. Generates detailed parameter difference reports.

**Significance:** Enables nuanced, configurable workflow similarity scoring. Core logic for determining how "different" two workflows are based on their structural and parameter changes.
