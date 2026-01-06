# File: `libs/langchain/langchain_classic/output_parsers/list.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 13 |
| Imports | langchain_core |

## Understanding

**Status:** ✅ Explored

**Purpose:** Re-exports list parsing utilities from langchain_core for backward compatibility.

**Mechanism:** Imports and re-exports four list parsers from langchain_core: CommaSeparatedListOutputParser, ListOutputParser, MarkdownListOutputParser, and NumberedListOutputParser.

**Significance:** Compatibility shim maintaining classic import paths for list parsing functionality. Essential for existing code using various list extraction patterns from LLM outputs.
