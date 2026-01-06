# File: `libs/langchain/langchain_classic/globals.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 19 |
| Imports | langchain_core |

## Understanding

**Status:** ✅ Explored

**Purpose:** Re-exports global configuration functions from langchain_core.

**Mechanism:** Simple import-and-export of debug mode, LLM cache, and verbosity control functions from `langchain_core.globals`: `get_debug`, `set_debug`, `get_verbose`, `set_verbose`, `get_llm_cache`, and `set_llm_cache`.

**Significance:** Provides a stable public API for global LangChain configuration while delegating implementation to langchain_core. These global settings affect behavior across all LangChain components (enabling debug logging, caching LLM responses, and verbose output).
