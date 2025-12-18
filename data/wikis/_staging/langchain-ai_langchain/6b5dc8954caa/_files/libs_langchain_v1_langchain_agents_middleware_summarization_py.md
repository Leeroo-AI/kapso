# File: `libs/langchain_v1/langchain/agents/middleware/summarization.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 535 |
| Classes | `SummarizationMiddleware` |
| Imports | collections, functools, langchain, langchain_core, langgraph, typing, typing_extensions, uuid, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Automatically summarizes conversation history when token/message limits are approached to prevent context overflow.

**Mechanism:** Implements SummarizationMiddleware that hooks into before_model. Uses token counter (tuned per model type - 3.3 chars/token for Anthropic) to check if trigger conditions are met (fraction of max tokens, absolute token count, or message count). Determines cutoff index using binary search for token-based limits or message counting. Preserves recent messages and AI/Tool pairs (never splits tool calls from their responses). Invokes summarization model with DEFAULT_SUMMARY_PROMPT to generate context summary. Replaces old messages with single HumanMessage containing summary using RemoveMessage(REMOVE_ALL_MESSAGES).

**Significance:** Critical middleware for long-running agent conversations that exceed model context limits. Enables agents to maintain conversation continuity while staying within token budgets. Supports flexible trigger conditions (ContextSize tuples) and intelligent message partitioning to preserve important conversation structure.
