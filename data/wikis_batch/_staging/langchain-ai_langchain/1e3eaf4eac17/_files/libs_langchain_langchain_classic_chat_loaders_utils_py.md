# File: `libs/langchain/langchain_classic/chat_loaders/utils.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 36 |
| Imports | langchain_classic, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides deprecated import shims for chat loader utility functions.

**Mechanism:** Uses dynamic import mechanism via create_importer to lazily load utility functions (map_ai_messages, map_ai_messages_in_session, merge_chat_runs, merge_chat_runs_in_session) from langchain_community.chat_loaders.utils when accessed.

**Significance:** Maintains backwards compatibility for chat processing utilities by redirecting deprecated imports from langchain_classic to their new location in langchain_community, supporting transformation and merging of chat session data.
