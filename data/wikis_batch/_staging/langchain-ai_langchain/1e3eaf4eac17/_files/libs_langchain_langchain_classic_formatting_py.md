# File: `libs/langchain/langchain_classic/formatting.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 5 |
| Imports | langchain_core |

## Understanding

**Status:** ✅ Explored

**Purpose:** Deprecated re-export of text formatting utilities

**Mechanism:** Re-exports `StrictFormatter` and `formatter` from `langchain_core.utils.formatting` to maintain backwards compatibility with code importing from the old location.

**Significance:** Deprecated compatibility layer allowing existing code to continue working while the canonical implementation has moved to langchain-core, the foundational layer of the LangChain architecture.
