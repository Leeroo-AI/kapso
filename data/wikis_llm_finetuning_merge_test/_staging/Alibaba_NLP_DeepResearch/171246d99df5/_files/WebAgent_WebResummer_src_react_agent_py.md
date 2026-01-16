# File: `WebAgent/WebResummer/src/react_agent.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 202 |
| Classes | `MultiTurnReactAgent` |
| Imports | json, openai, os, qwen_agent, summary_utils, tiktoken, transformers, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements the core ReAct (Reasoning + Acting) agent for multi-turn web research with conversation summarization support.

**Mechanism:** The `MultiTurnReactAgent` class extends Qwen Agent's `FnCallAgent`. It maintains conversation state and executes a think-act-observe loop: (1) calls an LLM server (vLLM via OpenAI-compatible API) to generate responses with tool calls or answers; (2) parses `<tool_call>` tags to invoke tools (search/visit); (3) appends tool responses and continues until `<answer>` is found or limits are reached. Key features include token counting (via AutoTokenizer or tiktoken), configurable context limits (MAX_CONTEXT), and the ReSum mechanism - when enabled (RESUM env var) or at specified iterations, it calls `summarize_conversation` to compress conversation history into a summary, resetting context while preserving essential information. Handles termination conditions: answer found, token limit exceeded, or LLM call limit reached.

**Significance:** The central reasoning engine of WebResummer. Implements the novel ReSum (Resummarization) approach for handling long-context web research tasks, enabling the agent to process extensive information while staying within context limits through intelligent summarization.
