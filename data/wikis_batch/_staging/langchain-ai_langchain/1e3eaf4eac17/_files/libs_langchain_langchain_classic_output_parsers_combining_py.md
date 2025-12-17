# File: `libs/langchain/langchain_classic/output_parsers/combining.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 58 |
| Classes | `CombiningOutputParser` |
| Imports | __future__, langchain_core, typing, typing_extensions |

## Understanding

**Status:** ✅ Explored

**Purpose:** Combine multiple output parsers to extract different structured data types from a single LLM response.

**Mechanism:** Takes a list of at least two parsers, generates combined format instructions with double-newline separators, splits LLM output by "\n\n", and applies each parser to its corresponding text segment. Validates that parsers cannot be nested combining parsers or list parsers, and merges all parsed results into a single dictionary.

**Significance:** Enables complex multi-format parsing scenarios where an LLM needs to produce multiple types of structured output in one response, such as generating both JSON metadata and a formatted list in a single call.
