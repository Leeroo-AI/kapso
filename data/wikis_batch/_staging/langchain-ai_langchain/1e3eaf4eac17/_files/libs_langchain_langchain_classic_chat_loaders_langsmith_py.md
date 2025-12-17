# File: `libs/langchain/langchain_classic/chat_loaders/langsmith.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 30 |
| Imports | langchain_classic, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides deprecated import shims for LangSmith chat loader classes.

**Mechanism:** Uses dynamic import mechanism via create_importer to lazily load LangSmithRunChatLoader and LangSmithDatasetChatLoader from langchain_community.chat_loaders.langsmith when accessed, implementing __getattr__ for attribute lookup.

**Significance:** Maintains backwards compatibility for LangSmith integration by redirecting deprecated imports from langchain_classic to their new location in langchain_community, supporting loading of chat data from LangSmith runs and datasets.
