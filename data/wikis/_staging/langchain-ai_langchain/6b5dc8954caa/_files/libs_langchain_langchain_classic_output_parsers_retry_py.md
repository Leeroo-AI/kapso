# File: `libs/langchain/langchain_classic/output_parsers/retry.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 315 |
| Classes | `RetryOutputParserRetryChainInput`, `RetryWithErrorOutputParserRetryChainInput`, `RetryOutputParser`, `RetryWithErrorOutputParser` |
| Imports | __future__, langchain_core, pydantic, typing, typing_extensions |

## Understanding

**Status:** ✅ Explored

**Purpose:** Retries parsing failures by sending original prompt and failed completion to LLM for regeneration.

**Mechanism:** Implements two variants: RetryOutputParser (sends prompt + completion) and RetryWithErrorOutputParser (adds error details). On OutputParserException, invokes retry_chain up to max_retries. Requires parse_with_prompt method with PromptValue. Supports sync/async and legacy/modern chain interfaces.

**Significance:** Critical error recovery pattern that uses context (original prompt) to guide LLM corrections. More sophisticated than OutputFixingParser as it provides full prompt context, enabling better understanding of requirements.
