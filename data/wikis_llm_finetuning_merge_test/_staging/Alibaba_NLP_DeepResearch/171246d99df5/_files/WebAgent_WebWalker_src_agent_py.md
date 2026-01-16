# File: `WebAgent/WebWalker/src/agent.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 208 |
| Classes | `WebWalker` |
| Imports | json, openai, prompts, qwen_agent, time, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements the WebWalker agent that navigates websites by clicking buttons/links to find information, using a ReAct-style approach with memory accumulation and progressive answer generation.

**Mechanism:** The `WebWalker` class extends Qwen Agent's `FnCallAgent` and implements website exploration. The `_run()` method uses a Thought/Action/Action Input/Observation loop where the agent reasons about which buttons to click. Key features include: (1) `observation_information_extraction()` - uses LLM to extract useful information from page observations; (2) `critic_information()` - evaluates if accumulated memory is sufficient to answer the query; (3) Memory accumulation in `self.momery` list that persists across steps; (4) `_detect_tool()` parses ReAct format to extract Action and Action Input; (5) `_prepend_react_prompt()` formats the system prompt with tool descriptions. The agent connects to OpenAI-compatible APIs (DashScope or OpenAI) for LLM calls with exponential backoff retry.

**Significance:** Core agent component for WebWalker that enables structured website navigation. Unlike WebSailor's open web search, WebWalker focuses on deep exploration within a single website by simulating button clicks and page navigation.
