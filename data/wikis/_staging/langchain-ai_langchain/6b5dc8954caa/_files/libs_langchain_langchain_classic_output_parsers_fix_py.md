# File: `libs/langchain/langchain_classic/output_parsers/fix.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 156 |
| Classes | `OutputFixingParserRetryChainInput`, `OutputFixingParser` |
| Imports | __future__, langchain_classic, langchain_core, pydantic, typing, typing_extensions |

## Understanding

**Status:** ✅ Explored

**Purpose:** Automatically fixes parsing errors by using an LLM to correct malformed outputs.

**Mechanism:** Wraps any BaseOutputParser and catches OutputParserException. On failure, invokes retry_chain (LLM + prompt) with original instructions, completion, and error details to generate corrected output. Supports configurable max_retries and both sync/async operations. Handles legacy LLMChain and modern Runnable interfaces.

**Significance:** Critical error recovery mechanism that makes parsing more robust by leveraging LLM intelligence to self-correct formatting mistakes, reducing parsing failures in production systems.
