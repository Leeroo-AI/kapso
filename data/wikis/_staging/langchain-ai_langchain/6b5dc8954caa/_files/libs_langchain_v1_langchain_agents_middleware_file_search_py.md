# File: `libs/langchain_v1/langchain/agents/middleware/file_search.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 387 |
| Classes | `FilesystemFileSearchMiddleware` |
| Imports | __future__, contextlib, datetime, fnmatch, json, langchain, langchain_core, pathlib, re, subprocess, ... +1 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides filesystem search tools (Glob and Grep) to agents through middleware, enabling pattern-based file discovery and content search within a sandboxed root directory.

**Mechanism:** FilesystemFileSearchMiddleware creates two @tool-decorated closures: glob_search uses pathlib.glob for pattern matching (e.g., **/*.js) returning sorted file paths with modification times, grep_search performs regex content search preferring ripgrep --json subprocess when available (30s timeout) with Python regex fallback, supporting include filters (brace expansion *.{py,pyi}), three output modes (files_with_matches/content/count), and max_file_size_mb limits. Both tools validate and resolve virtual paths (/) to filesystem paths within root_path using path traversal checks (.., ~), converting results back to virtual paths.

**Significance:** Critical tooling infrastructure for code-aware agents - provides the same search primitives used by Claude Code and similar systems, enabling agents to efficiently explore large codebases and locate relevant files/content without exposing the entire filesystem or requiring manual file listing.
