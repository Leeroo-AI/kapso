# File: `libs/langchain/langchain_classic/chat_loaders/gmail.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 23 |
| Imports | langchain_classic, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides deprecated import shim for Gmail chat loader class.

**Mechanism:** Uses dynamic import mechanism via create_importer to lazily load GMailLoader from langchain_community.chat_loaders.gmail when accessed, implementing __getattr__ for attribute lookup.

**Significance:** Maintains backwards compatibility for Gmail integration by redirecting deprecated imports from langchain_classic to their new location in langchain_community, supporting legacy codebases during migration.
