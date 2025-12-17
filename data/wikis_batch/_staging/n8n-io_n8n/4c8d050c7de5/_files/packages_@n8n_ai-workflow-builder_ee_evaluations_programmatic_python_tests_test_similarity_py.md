# File: `packages/@n8n/ai-workflow-builder.ee/evaluations/programmatic/python/tests/test_similarity.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 328 |
| Functions | `test_identical_workflows`, `test_empty_workflows`, `test_missing_node`, `test_trigger_mismatch`, `test_parameter_differences`, `test_connection_difference`, `test_trigger_parameter_update_priority`, `test_trigger_deletion_is_critical`, `... +1 more` |
| Imports | src |

## Understanding

**Status:** ✅ Explored

**Purpose:** Unit tests for similarity calculation

**Mechanism:** Comprehensive test suite for graph edit distance calculation covering: identical workflows (100% similarity), empty workflows, missing/extra nodes (insertion edits), trigger type mismatches (high cost, critical priority), parameter differences (moderate similarity impact), connection differences (edge operations), trigger parameter updates (should be minor not critical), trigger deletions (should be critical), and trigger insertions (should be critical). Tests validate similarity scores, edit costs, operation types, and priority levels. Uses both standard config and preset configurations.

**Significance:** Validates the core similarity algorithm that evaluates AI-generated workflows. These tests ensure the system correctly prioritizes different types of changes (triggers are critical, parameters are minor) and produces meaningful similarity scores. Essential for maintaining confidence in the AI workflow builder evaluation system and catching regressions in the complex cost function logic.
