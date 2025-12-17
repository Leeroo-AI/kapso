# File: `libs/langchain/langchain_classic/globals.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 19 |
| Imports | langchain_core |

## Understanding

**Status:** ✅ Explored

**Purpose:** Re-exports global configuration functions from langchain_core.globals to maintain backwards compatibility with legacy import paths.

**Mechanism:** Directly imports six functions (get_debug, get_llm_cache, get_verbose, set_debug, set_llm_cache, set_verbose) from langchain_core.globals and lists them in `__all__` for public API exposure, creating a transparent pass-through to the core implementation.

**Significance:** Preserves the public API surface for global LangChain configuration settings (debug mode, LLM caching, verbosity) that were previously accessible from langchain_classic, allowing existing codebases to continue using these settings without updating import statements during the migration to the modular architecture.
