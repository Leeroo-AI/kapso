# File: `libs/langchain_v1/langchain/agents/middleware/file_search.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 387 |
| Classes | `FilesystemFileSearchMiddleware` |
| Imports | __future__, contextlib, datetime, fnmatch, json, langchain, langchain_core, pathlib, re, subprocess, ... +1 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides filesystem search capabilities to agents through `Glob` and `Grep` tools that operate on a sandboxed root directory, enabling agents to discover and search files programmatically.

**Mechanism:** The middleware registers two LangChain tools as closures capturing the configured `root_path`: (1) `glob_search()` uses pathlib's `glob()` method to match file patterns (e.g., `**/*.py`), returning virtual paths relative to root sorted by modification time; (2) `grep_search()` performs content search with three output modes (files_with_matches, content with line numbers, count). The grep implementation prefers `rg` (ripgrep) for performance, parsing its JSON output format, but falls back to pure Python regex search if ripgrep is unavailable or times out. Path validation prevents traversal attacks by checking for `..` and `~`, resolving paths, and ensuring they remain within the root directory boundary using `Path.relative_to()`. Include patterns support brace expansion (`*.{py,pyi}`) via custom `_expand_include_patterns()` logic.

**Significance:** This middleware enables code-aware agents that need to explore codebases, search for patterns, or analyze file structures. It's analogous to giving agents filesystem read access but with strict security boundaries. The ripgrep integration provides production-grade performance for large codebases (millions of lines), while the Python fallback ensures portability. The virtual path system (all paths start with `/`) creates a clean, sandboxed abstraction that prevents accidental or malicious access to files outside the workspace. This is commonly used with Anthropic's text editor and memory tools patterns.
