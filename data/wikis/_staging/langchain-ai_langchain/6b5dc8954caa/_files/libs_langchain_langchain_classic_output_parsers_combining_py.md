# File: `libs/langchain/langchain_classic/output_parsers/combining.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 58 |
| Classes | `CombiningOutputParser` |
| Imports | __future__, langchain_core, typing, typing_extensions |

## Understanding

**Status:** ✅ Explored

**Purpose:** Combines multiple output parsers to extract different structured outputs from a single LLM response.

**Mechanism:** Splits LLM output on double newlines and applies each parser sequentially to corresponding sections. Validates at least 2 parsers, prevents nesting combining parsers, and prohibits list parsers. Aggregates results into a single dictionary by updating with each parser's output.

**Significance:** Enables complex multi-format output extraction from single LLM calls, reducing API costs and latency by consolidating multiple parsing operations. Useful when requesting multiple data structures in one prompt.
