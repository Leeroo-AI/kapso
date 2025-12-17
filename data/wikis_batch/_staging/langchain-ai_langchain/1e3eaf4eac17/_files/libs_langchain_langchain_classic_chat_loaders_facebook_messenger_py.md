# File: `libs/langchain/langchain_classic/chat_loaders/facebook_messenger.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 32 |
| Imports | langchain_classic, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides deprecated import shims for Facebook Messenger chat loader classes.

**Mechanism:** Uses dynamic import mechanism via create_importer to lazily load SingleFileFacebookMessengerChatLoader and FolderFacebookMessengerChatLoader from langchain_community when accessed, implementing __getattr__ for attribute lookup.

**Significance:** Maintains backwards compatibility for Facebook Messenger integration by redirecting deprecated imports from langchain_classic to their new location in langchain_community, enabling gradual migration.
