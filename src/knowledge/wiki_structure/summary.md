# Wiki Structure Summary

This document summarizes the architecture of the Knowledge Graph, designed to be a "Single Source of Truth" for executable ML/Data knowledge.

## 1. Node Types (The Entities)

| Node Type | Role | Key Constraint |
|:---|:---|:---|
| **Workflow** | The Recipe | Ordered sequence of steps. |
| **Principle** | The Theory | **MUST** be executable (link to Implementation). |
| **Implementation** | The Code | Source of truth for syntax/execution. |
| **Environment** | The Context | Leaf node (Prerequisites). |
| **Heuristic** | The Wisdom | Leaf node (Optimizations/Tips). |

## 2. Connections (The Skeleton)

The graph follows a strict **Top-Down Directed Acyclic Graph (DAG)** schema.

*   **Workflow** `step` → **Principle**
*   **Workflow** `uses_heuristic` → **Heuristic**
*   **Principle** `implemented_by` → **Implementation**
*   **Principle** `uses_heuristic` → **Heuristic**
*   **Implementation** `requires_env` → **Environment**
*   **Implementation** `uses_heuristic` → **Heuristic**

## 3. Composers (The Brain)

Dynamic functions that generate views on top of the static graph:

*   **Hierarchy Composer:** Groups related Principles by creating a new parent Principle (e.g., `Adam` + `SGD` → merged into new `Optimization` node).
*   **Workflow Composer:** Chains Principles based on I/O compatibility to form Workflows.

## Files Reference

*   `page_connections.md`: Detailed edge registry and rules.
*   `composers.md`: Logic for dynamic structure generation.
*   `node_*.md`: Definition and templates for each node type.

