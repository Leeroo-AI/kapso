# File: `libs/langchain/langchain_classic/sql_database.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 25 |
| Imports | langchain_classic, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Backwards compatibility shim for deprecated SQLDatabase utility.

**Mechanism:** Uses create_importer to lazily redirect SQLDatabase imports to langchain_community.utilities, issuing deprecation warnings through the __getattr__ mechanism. Exports in __all__ for backward compatibility.

**Significance:** Maintains backward compatibility for legacy SQL database wrapper moved to langchain_community, allowing existing SQL agent and chain code to continue working during migration period.
