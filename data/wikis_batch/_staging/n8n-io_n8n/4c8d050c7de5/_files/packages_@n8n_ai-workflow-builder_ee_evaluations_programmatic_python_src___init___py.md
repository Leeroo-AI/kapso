# File: `packages/@n8n/ai-workflow-builder.ee/evaluations/programmatic/python/src/__init__.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 21 |
| Imports | __future__, src |

## Understanding

**Status:** ✅ Explored

**Purpose:** Package initialization and public API definition

**Mechanism:** Exports the main public interface of the workflow comparison module including WorkflowComparisonConfig, load_config, build_workflow_graph, graph_stats, and calculate_graph_edit_distance. Sets version to 0.1.0 and uses __all__ to control exported symbols for clean imports.

**Significance:** Provides a clean entry point for the n8n workflow comparison module. This is the main interface that external code uses to access graph-based workflow similarity comparison functionality using NetworkX.
