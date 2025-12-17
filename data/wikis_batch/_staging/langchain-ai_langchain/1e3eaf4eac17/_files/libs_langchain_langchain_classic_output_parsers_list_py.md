# File: `libs/langchain/langchain_classic/output_parsers/list.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 13 |
| Imports | langchain_core |

## Understanding

**Status:** ✅ Explored

**Purpose:** Re-export list parsing classes from langchain_core for parsing various list formats from LLM output.

**Mechanism:** Imports and exposes CommaSeparatedListOutputParser, ListOutputParser, MarkdownListOutputParser, and NumberedListOutputParser from langchain_core.output_parsers.list, providing unified access to different list parsing strategies.

**Significance:** Provides a convenience import location for list-based output parsers, enabling developers to parse LLM outputs formatted as comma-separated values, markdown lists, or numbered lists without needing to import from langchain_core directly.
