# File: `libs/langchain/langchain_classic/chat_loaders/base.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 3 |
| Imports | langchain_core |

## Understanding

**Status:** ✅ Explored

**Purpose:** Re-exports the base chat loader abstraction from langchain-core for backward compatibility.

**Mechanism:** Imports `BaseChatLoader` from `langchain_core.chat_loaders` and exposes it through the `__all__` list, making it available at the langchain_classic package level.

**Significance:** Maintains API compatibility by providing access to the core `BaseChatLoader` interface from the legacy langchain_classic package, allowing existing code to continue using the base abstraction without directly importing from langchain-core.
