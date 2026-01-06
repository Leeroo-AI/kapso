# File: `packages/@n8n/ai-workflow-builder.ee/evaluations/programmatic/python/tests/test_similarity.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 328 |
| Functions | `test_identical_workflows`, `test_empty_workflows`, `test_missing_node`, `test_trigger_mismatch`, `test_parameter_differences`, `test_connection_difference`, `test_trigger_parameter_update_priority`, `test_trigger_deletion_is_critical`, `... +1 more` |
| Imports | src |

## Understanding

**Status:** ✅ Explored

**Purpose:** Test workflow similarity calculation

**Mechanism:** Unit tests for graph edit distance and similarity scoring covering identical workflows, empty workflows, node insertion/deletion, trigger mismatches, parameter differences, connection differences, and edit operation priority determination.

**Significance:** Validates complex similarity calculation logic. Ensures accurate workflow comparison for AI evaluation purposes.
