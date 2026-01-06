# File: `libs/langchain/langchain_classic/chat_models/tongyi.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 23 |
| Imports | langchain_classic, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides backward-compatible imports for Alibaba Tongyi chat model from langchain_community.

**Mechanism:** Uses the `create_importer` utility with a `DEPRECATED_LOOKUP` dictionary mapping `ChatTongyi` to its langchain_community location. Implements `__getattr__` for dynamic attribute resolution with deprecation warnings.

**Significance:** Part of the deprecation migration strategy for langchain_classic, redirecting imports to the actual Alibaba Tongyi chat model implementation in langchain_community while maintaining backward compatibility for users still importing from the old location.
