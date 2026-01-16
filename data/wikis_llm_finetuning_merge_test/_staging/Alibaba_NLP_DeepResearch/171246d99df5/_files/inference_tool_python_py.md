# File: `inference/tool_python.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 150 |
| Classes | `PythonInterpreter` |
| Functions | `has_chinese_chars` |
| Imports | concurrent, json5, os, qwen_agent, random, re, requests, sandbox_fusion, time, typing |

## Understanding

**Status:** Explored

**Purpose:** Sandboxed Python code execution tool that allows the DeepResearch agent to run Python code safely, enabling data analysis, calculations, and programmatic operations during research.

**Mechanism:** The `PythonInterpreter` class uses the `sandbox_fusion` library to execute Python code in isolated environments. Code is submitted to configurable sandbox endpoints (from `SANDBOX_FUSION_ENDPOINT` env var) with retry logic across up to 8 attempts, randomly sampling from available endpoints. Handles timeouts, captures stdout/stderr, and returns execution results. The tool expects code within `<code></code>` XML tags and emphasizes using `print()` for output visibility.

**Significance:** Critical capability tool that enables computational reasoning. Allows the agent to perform data processing, numerical calculations, web scraping, and other programmatic tasks that would be difficult or impossible through pure language reasoning. The sandboxed execution ensures security when running agent-generated code.
