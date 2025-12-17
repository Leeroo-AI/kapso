# File: `libs/langchain/langchain_classic/text_splitter.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 50 |
| Imports | langchain_text_splitters |

## Understanding

**Status:** ✅ Explored

**Purpose:** Backwards compatibility re-export of text splitting utilities

**Mechanism:** Re-exports 19 text splitter classes and utilities from the dedicated `langchain_text_splitters` package, including character-based, recursive, HTML, JSON, Markdown, code-aware, and NLP-based splitters.

**Significance:** Compatibility layer for document chunking utilities that have been extracted into a separate `langchain_text_splitters` package. Maintains the old import path to support legacy code while the actual implementations live in their own dedicated package for better modularity.
