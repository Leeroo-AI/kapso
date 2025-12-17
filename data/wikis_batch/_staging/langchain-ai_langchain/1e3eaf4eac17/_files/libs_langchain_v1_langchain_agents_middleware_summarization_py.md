# File: `libs/langchain_v1/langchain/agents/middleware/summarization.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 535 |
| Classes | `SummarizationMiddleware` |
| Imports | collections, functools, langchain, langchain_core, langgraph, typing, typing_extensions, uuid, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Automatically summarizes conversation history when token or message limits are approached, preventing context window overflow while maintaining continuity by preserving recent messages and ensuring AI/Tool message pairs remain together.

**Mechanism:** The SummarizationMiddleware operates in the `before_model` hook, checking message counts and token usage against configurable triggers (fraction of model's max tokens, absolute token count, or message count). When triggered, it partitions messages into those to summarize and those to keep (using binary search for token-based cutoffs), ensures AI/Tool pairs aren't split, generates a summary using an LLM with a configurable prompt, and replaces old messages with a HumanMessage containing the summary. The middleware uses approximate token counting tuned per model type, trims messages before summarization to avoid recursive overflow, and assigns UUIDs to messages for the add_messages reducer.

**Significance:** This middleware is essential for long-running agent conversations that would otherwise exceed model context limits. It enables agents to maintain context over extended interactions (customer service sessions, multi-day coding projects) without losing critical information. The flexible trigger system (supports multiple conditions with OR logic) and retention policies (keep by tokens, messages, or fraction) make it adaptable to different models and use cases. The careful handling of AI/Tool pairs ensures tool execution context isn't corrupted, which is crucial for agent reliability.
