# File: `libs/langchain/langchain_classic/chains/prompt_selector.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 65 |
| Classes | `BasePromptSelector`, `ConditionalPromptSelector` |
| Functions | `is_llm`, `is_chat_model` |
| Imports | abc, collections, langchain_core, pydantic |

## Understanding

**Status:** ✅ Explored

**Purpose:** Enables dynamic prompt selection based on the type or characteristics of a language model.

**Mechanism:** `BasePromptSelector` defines abstract interface with `get_prompt` method. `ConditionalPromptSelector` implements this by iterating through condition-prompt tuples, returning the first matching prompt or a default. Helper functions `is_llm` and `is_chat_model` provide common conditionals for distinguishing model types.

**Significance:** Allows chains and applications to adapt prompts based on the model being used (e.g., different prompts for chat models vs completion models). Essential for building model-agnostic chains that work correctly across different LLM types without hardcoding model-specific logic.
