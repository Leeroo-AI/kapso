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

**Purpose:** Provides classes for dynamically selecting prompts based on language model type.

**Mechanism:** Defines `BasePromptSelector` abstract interface and `ConditionalPromptSelector` implementation that evaluates a list of condition-prompt tuples, returning the first matching prompt or falling back to a default. Includes helper functions `is_llm` and `is_chat_model` for common conditionals.

**Significance:** Utility for prompt adaptation across different model types. Allows applications to use chat-formatted prompts for chat models (ChatGPT) and completion-style prompts for traditional LLMs (GPT-3), improving compatibility without manual model type checking.
