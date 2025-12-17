# File: `libs/langchain/langchain_classic/chat_loaders/imessage.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 23 |
| Imports | langchain_classic, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides deprecated import shim for iMessage chat loader class.

**Mechanism:** Uses dynamic import mechanism via create_importer to lazily load IMessageChatLoader from langchain_community.chat_loaders.imessage when accessed, implementing __getattr__ for attribute lookup.

**Significance:** Maintains backwards compatibility for iMessage integration by redirecting deprecated imports from langchain_classic to their new location in langchain_community, enabling smooth package structure transitions.
