# File: `libs/langchain/langchain_classic/output_parsers/prompts.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 21 |
| Imports | langchain_core |

## Understanding

**Status:** ✅ Explored

**Purpose:** Define prompt templates for retry and fixing output parsers to guide LLMs in correcting malformed outputs.

**Mechanism:** Defines NAIVE_FIX string template with placeholders for instructions, completion, and error, then creates NAIVE_FIX_PROMPT PromptTemplate from the string. The template explains that a completion failed to satisfy constraints and asks the LLM to try again.

**Significance:** Provides the prompt template used by OutputFixingParser to request corrected output from an LLM when initial parsing fails, enabling self-correcting parsing workflows through clear error communication.
