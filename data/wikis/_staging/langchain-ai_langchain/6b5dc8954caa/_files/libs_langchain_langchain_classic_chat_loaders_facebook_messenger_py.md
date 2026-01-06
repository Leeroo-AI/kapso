# File: `libs/langchain/langchain_classic/chat_loaders/facebook_messenger.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 32 |
| Imports | langchain_classic, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides backward-compatible imports for Facebook Messenger chat loaders from langchain_community.

**Mechanism:** Uses the `create_importer` utility with a `module_lookup` dictionary to lazily import `SingleFileFacebookMessengerChatLoader` and `FolderFacebookMessengerChatLoader` from langchain_community. Implements `__getattr__` for dynamic attribute lookup with deprecation warnings.

**Significance:** Part of the deprecation migration strategy for langchain_classic, redirecting imports to the actual implementations in langchain_community while maintaining backward compatibility for users still importing from the old location.
