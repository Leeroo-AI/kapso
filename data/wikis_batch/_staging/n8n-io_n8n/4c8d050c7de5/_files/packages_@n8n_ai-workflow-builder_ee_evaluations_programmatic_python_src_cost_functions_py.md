# File: `packages/@n8n/ai-workflow-builder.ee/evaluations/programmatic/python/src/cost_functions.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 497 |
| Functions | `normalize_expression`, `node_substitution_cost`, `node_deletion_cost`, `node_insertion_cost`, `edge_substitution_cost`, `edge_deletion_cost`, `edge_insertion_cost`, `compare_parameters`, `... +5 more` |
| Imports | re, src, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Cost calculation functions for graph edit distance operations

**Mechanism:** Implements configuration-aware cost functions for all graph edit operations. Key functions: normalize_expression() removes comments and whitespace and normalizes $fromAI() calls; node_substitution_cost() calculates costs based on type similarity, trigger status, parameter differences, and name hash mismatches; compare_parameters() performs deep recursive comparison with rule-based handling (semantic, numeric tolerance, normalized); get_parameter_diff() extracts detailed diffs for display. Uses special handling for trigger nodes (highest cost), same-type substitutions (parameter-based cost), similar types (medium cost), and different types (high cost). Includes helper functions for semantic similarity (Jaccard), list comparison, and value normalization.

**Significance:** The intelligence behind workflow comparison. These cost functions encode domain knowledge about what differences matter most in n8n workflows. Trigger mismatches are treated as critical, parameter variations are weighted appropriately, and the system can distinguish between minor formatting differences vs substantive changes. Essential for producing meaningful similarity scores that reflect actual workflow quality.
