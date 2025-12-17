# File: `packages/@n8n/ai-workflow-builder.ee/evaluations/programmatic/python/tests/test_graph_builder.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 161 |
| Functions | `test_build_simple_workflow_graph`, `test_build_graph_with_filtering`, `test_parameter_filtering`, `test_is_trigger_node`, `test_graph_stats`, `test_empty_workflow` |
| Imports | src |

## Understanding

**Status:** ✅ Explored

**Purpose:** Unit tests for graph builder module

**Mechanism:** Tests graph construction from n8n workflow JSON including: building simple graphs with nodes and edges (test_build_simple_workflow_graph), filtering ignored node types like sticky notes (test_build_graph_with_filtering), filtering parameters by global ignore rules (test_parameter_filtering), trigger node detection logic (test_is_trigger_node), graph statistics calculation (test_graph_stats), and empty workflow handling (test_empty_workflow). Uses inline workflow JSON fixtures and assertion-based validation.

**Significance:** Ensures the graph builder correctly transforms n8n workflows into NetworkX graphs with proper filtering and metadata. Critical for maintaining reliability of the workflow comparison pipeline since all downstream analysis depends on correct graph construction.
