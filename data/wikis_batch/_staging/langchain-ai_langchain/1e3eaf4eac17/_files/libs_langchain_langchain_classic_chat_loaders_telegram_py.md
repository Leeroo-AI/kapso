# File: `libs/langchain/langchain_classic/chat_loaders/telegram.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 23 |
| Imports | langchain_classic, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides deprecated import shim for Telegram chat loader class.

**Mechanism:** Uses dynamic import mechanism via create_importer to lazily load TelegramChatLoader from langchain_community.chat_loaders.telegram when accessed, implementing __getattr__ for attribute lookup.

**Significance:** Maintains backwards compatibility for Telegram integration by redirecting deprecated imports from langchain_classic to their new location in langchain_community, enabling loading of Telegram chat messages.
