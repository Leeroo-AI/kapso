# File: `libs/langchain/langchain_classic/sql_database.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 25 |
| Imports | langchain_classic, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Backwards compatibility proxy for SQL database utility

**Mechanism:** Uses dynamic attribute lookup via `__getattr__` and `create_importer` to lazily import `SQLDatabase` from `langchain_community.utilities` when accessed, providing deprecation warnings.

**Significance:** Compatibility layer for the SQL database wrapper that has been moved to langchain-community. Preserves the legacy import path to avoid breaking existing code while directing users to the new canonical location for database utilities.
