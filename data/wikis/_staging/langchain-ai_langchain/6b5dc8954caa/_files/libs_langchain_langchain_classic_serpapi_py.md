# File: `libs/langchain/langchain_classic/serpapi.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 25 |
| Imports | langchain_classic, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Backwards compatibility shim for deprecated SerpAPI search wrapper.

**Mechanism:** Uses create_importer to lazily redirect SerpAPIWrapper imports to langchain_community.utilities, issuing deprecation warnings through the __getattr__ mechanism. Exports in __all__ for backward compatibility.

**Significance:** Maintains backward compatibility for legacy SerpAPI integration moved to langchain_community, supporting existing code while encouraging migration to the new location.
