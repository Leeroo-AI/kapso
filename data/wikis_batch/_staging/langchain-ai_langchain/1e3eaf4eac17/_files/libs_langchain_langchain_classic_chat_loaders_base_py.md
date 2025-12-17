# File: `libs/langchain/langchain_classic/chat_loaders/base.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 3 |
| Imports | langchain_core |

## Understanding

**Status:** ✅ Explored

**Purpose:** Re-exports the BaseChatLoader abstract interface from langchain_core.

**Mechanism:** Imports BaseChatLoader from langchain_core.chat_loaders and exposes it via __all__ for external use, providing a stable import path in langchain_classic.

**Significance:** Maintains backwards compatibility by providing access to the core BaseChatLoader interface through the langchain_classic package, allowing existing code to continue using this import path.
