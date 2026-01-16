# File: `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/utils/code_safety_checker.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 345 |
| Classes | `CodeSafetyChecker`, `_DangerousCodeVisitor` |
| Functions | `check_banned_operations` |
| Imports | ast, io, tokenize |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** Explored

**Purpose:** Provides security validation for Python code execution by detecting potentially dangerous or banned operations before code is run by the agent.

**Mechanism:** Two-layer security approach:
1. `check_banned_operations()`: Uses Python's `tokenize` module to scan code for banned keywords (exit, yield, requests, url, pip, install, conda) at the token level, ignoring strings/comments to reduce false positives. Returns a tuple of (is_safe, message).
2. `CodeSafetyChecker` class: Uses AST (Abstract Syntax Tree) analysis via a nested `_DangerousCodeVisitor` class to detect dangerous function calls. Tracks imports/aliases and identifies:
   - File operations (os.remove, shutil.rmtree, etc.)
   - Subprocess calls (subprocess.run with shell=True flagged as especially dangerous)
   - Builtins (eval, exec, open with write modes)
   - Data write methods (to_csv, to_excel, to_json, to_parquet, to_pickle)

**Significance:** Critical security component for safe code execution in the agent framework. When the agent generates or receives Python code to execute (e.g., via code interpreter tools), this checker validates the code before execution to prevent malicious or unintended harmful operations like file deletion, arbitrary command execution, or package installation.
