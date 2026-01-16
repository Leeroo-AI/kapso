# File: `WebAgent/WebDancer/demos/agents/search_agent.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 113 |
| Classes | `SearchAgent` |
| Imports | copy, qwen_agent, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Defines the `SearchAgent` class, the core agent for web information seeking in WebDancer.

**Mechanism:** Extends `qwen_agent.agents.Assistant` to create a specialized search agent. Key features include: (1) `_run()` method that orchestrates LLM calls and tool execution in a loop, (2) custom system prompt injection via `make_system_prompt`, (3) custom user prompt prepending via `insert_in_custom_user_prompt()`, (4) reasoning mode support that wraps responses in `<think>` tags, (5) configurable max LLM calls (default 20), and (6) optional chaining to an additional agent for post-processing. The agent detects tool calls, executes them via `_call_tool()`, and appends results back to the conversation until no more tools are needed.

**Significance:** Core component of WebDancer. This is the main agent class that drives the web search and information gathering process, coordinating between LLM reasoning and tool execution (search/visit).
