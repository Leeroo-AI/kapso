# File: `libs/langchain/langchain_classic/output_parsers/retry.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 315 |
| Classes | `RetryOutputParserRetryChainInput`, `RetryWithErrorOutputParserRetryChainInput`, `RetryOutputParser`, `RetryWithErrorOutputParser` |
| Imports | __future__, langchain_core, pydantic, typing, typing_extensions |

## Understanding

**Status:** ✅ Explored

**Purpose:** Automatically retry parsing failures by re-prompting the LLM with the original prompt, failed completion, and optionally the error details.

**Mechanism:** Provides two parser variants - RetryOutputParser (sends prompt and completion) and RetryWithErrorOutputParser (also sends error details). Both wrap another parser, catch OutputParserException, invoke a retry chain (LLM) with context about the failure up to max_retries times, and support both sync/async parsing with legacy LLMChain and modern Runnable compatibility.

**Significance:** Enables resilient parsing through LLM self-correction by providing the original context when parsing fails, allowing the LLM to regenerate output that better matches the required format, with RetryWithErrorOutputParser providing more information for better corrections.
