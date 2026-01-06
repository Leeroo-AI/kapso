# File: `libs/langchain/langchain_classic/output_parsers/prompts.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 21 |
| Imports | langchain_core |

## Understanding

**Status:** ✅ Explored

**Purpose:** Defines the prompt template for OutputFixingParser to request LLM corrections.

**Mechanism:** Provides NAIVE_FIX template string with placeholders for instructions, completion, and error. Converts to PromptTemplate instance (NAIVE_FIX_PROMPT) for use in retry chains.

**Significance:** Single source of truth for the fixing prompt used by OutputFixingParser. The template structure guides the LLM to understand what went wrong and produce corrected output.
