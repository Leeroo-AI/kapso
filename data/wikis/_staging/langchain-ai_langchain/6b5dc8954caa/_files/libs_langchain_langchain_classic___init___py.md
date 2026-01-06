# File: `libs/langchain/langchain_classic/__init__.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 424 |
| Imports | importlib, langchain_core, typing, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Package entrypoint that manages deprecated imports and backwards compatibility for langchain_classic.

**Mechanism:** Uses a custom `__getattr__` function to intercept attribute access and dynamically import deprecated classes from their new locations (langchain_core, langchain_community, langchain_experimental). When accessed, each import triggers a deprecation warning via `_warn_on_import()` that guides users to the correct new import path. The file also calls `surface_langchain_deprecation_warnings()` on load to enable deprecation warnings.

**Significance:** Critical backwards compatibility layer. This is the main entry point for the langchain_classic package and ensures legacy code continues to work while warning users about deprecated import patterns. Maintains the transition path from old "import from root" patterns to the new modular structure (core/community/partners).
