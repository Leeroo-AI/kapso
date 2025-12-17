# File: `libs/langchain/langchain_classic/output_parsers/fix.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 156 |
| Classes | `OutputFixingParserRetryChainInput`, `OutputFixingParser` |
| Imports | __future__, langchain_classic, langchain_core, pydantic, typing, typing_extensions |

## Understanding

**Status:** ✅ Explored

**Purpose:** Automatically fix parsing errors by invoking an LLM to correct malformed output based on error messages and format instructions.

**Mechanism:** Wraps another parser and catches OutputParserException errors, then invokes a retry chain (typically an LLM) with the original instructions, failed completion, and error message to generate corrected output. Supports configurable max_retries, both sync and async parsing, and legacy LLMChain compatibility alongside modern Runnable chains.

**Significance:** Enables resilient parsing workflows where LLM outputs that don't initially match the expected format can be automatically corrected through self-reflection, significantly improving reliability when strict output formatting is required.
