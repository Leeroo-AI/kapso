# File: `packages/@n8n/task-runner-python/src/task_analyzer.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 212 |
| Classes | `SecurityValidator`, `TaskAnalyzer` |
| Imports | ast, collections, hashlib, src |

## Understanding

**Status:** ✅ Explored

**Purpose:** Code security analysis and validation

**Mechanism:** TaskAnalyzer parses Python code using AST to detect imports and analyze structure. SecurityValidator checks discovered imports against allowed/blocked lists. Uses AST walking to find all import statements including dynamic imports.

**Significance:** Pre-execution security layer. Validates user code before execution to catch blocked imports and potentially dangerous patterns.
