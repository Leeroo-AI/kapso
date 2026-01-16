# File: `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/tools/code_interpreter.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 413 |
| Classes | `CodeInterpreter`, `AnyThreadEventLoopPolicy` |
| Functions | `execute_with_timeout` |
| Imports | ast, asyncio, atexit, base64, concurrent, glob, io, json, os, pathlib, ... +12 more |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** Explored

**Purpose:** Implements a secure, sandboxed Python code execution environment using Jupyter kernels, enabling LLM agents to run Python code with visualization support and safety checks.

**Mechanism:** The `CodeInterpreter` class (registered as `'code_interpreter'`) manages Jupyter kernel processes: (1) `_start_kernel()` spawns a subprocess running IPython kernel with a connection file for IPC; (2) `_execute_code()` sends code to the kernel via `BlockingKernelClient` and collects outputs including text results, error tracebacks, and base64-encoded images; (3) `_serve_image()` saves matplotlib/PIL images to disk and returns URLs; (4) Security is enforced through `CodeSafetyChecker` AST analysis and `check_banned_operations()` before execution; (5) Timeout handling via `_M6CountdownTimer` and `execute_with_timeout()` with ThreadPoolExecutor; (6) CJK font support is patched via `_fix_matplotlib_cjk_font_issue()` copying AlibabaPuHuiTi font; (7) Kernel lifecycle management uses `atexit` handlers to clean up subprocesses. Uses `AnyThreadEventLoopPolicy` (borrowed from Tornado) for asyncio compatibility in non-main threads.

**Significance:** Critical tool that enables LLM agents to perform data analysis, mathematical computations, visualization generation, and general programming tasks. The security measures (AST checking, banned operations, sandboxing) make it suitable for production use with untrusted LLM-generated code. This is one of the most powerful capabilities in the agent toolkit.
