# File: `packages/@n8n/task-runner-python/src/task_analyzer.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 212 |
| Classes | `SecurityValidator`, `TaskAnalyzer` |
| Imports | ast, collections, hashlib, src |

## Understanding

**Status:** ✅ Explored

**Purpose:** Static security validation of Python code

**Mechanism:** Analyzes code before execution using AST (Abstract Syntax Tree) traversal:
1. SecurityValidator visits AST nodes to detect security violations
2. Validates import statements against stdlib_allow and external_allow lists
3. Blocks relative imports (e.g., from . import foo)
4. Blocks access to dangerous names (__loader__, __builtins__, __globals__, __spec__, __name__)
5. Blocks access to dangerous attributes (__subclasses__, __globals__, etc.)
6. Detects name-mangled attributes (_ClassName__attr pattern)
7. Identifies dynamic __import__() calls (security risk)
8. Caches validation results by code hash + allowlists for performance (LRU, max 500 entries)
9. Raises SecurityViolationError with detailed violation descriptions

**Significance:** This is the first line of defense in the security model, providing static analysis before code execution. The AST-based approach catches malicious patterns that could bypass subprocess security. The caching mechanism is crucial for performance when running similar code repeatedly. Works in conjunction with TaskExecutor's runtime security (import hooks, builtin filtering) for defense-in-depth.
