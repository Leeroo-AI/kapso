# File: `libs/langchain/langchain_classic/chat_loaders/langsmith.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 30 |
| Imports | langchain_classic, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides backward-compatible imports for LangSmith chat loaders from langchain_community.

**Mechanism:** Uses the `create_importer` utility with a `DEPRECATED_LOOKUP` dictionary mapping `LangSmithRunChatLoader` and `LangSmithDatasetChatLoader` to their langchain_community location. Implements `__getattr__` for dynamic attribute resolution with deprecation warnings.

**Significance:** Part of the deprecation migration strategy for langchain_classic, redirecting imports to the actual implementations in langchain_community while maintaining backward compatibility for users still importing from the old location. Supports loading chat sessions from LangSmith runs and datasets.
