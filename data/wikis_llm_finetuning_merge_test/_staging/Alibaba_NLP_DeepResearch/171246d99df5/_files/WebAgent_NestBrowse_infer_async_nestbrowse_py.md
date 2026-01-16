# File: `WebAgent/NestBrowse/infer_async_nestbrowse.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 203 |
| Functions | `call_tool`, `agentic_loop`, `main` |
| Imports | asyncio, collections, copy, json, os, prompts, random, re, toolkit, tqdm, ... +3 more |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Main inference engine for the NestBrowse web agent that conducts multi-turn, tool-augmented research conversations to answer complex questions.

**Mechanism:** Implements an asynchronous agentic loop (`agentic_loop`) that iteratively calls an LLM and executes tools (search, visit, click, fill) based on XML-formatted tool calls. Uses MCP (Model Context Protocol) for browser interaction, semaphores for concurrency control, and processes tool responses within `<tool_call>` and `<tool_response>` tags. The loop terminates when the agent produces an `<answer>` tag or exceeds token/turn limits. The `main` function manages rollout scheduling across a dataset with resume capability.

**Significance:** Core orchestration component of NestBrowse. It coordinates LLM reasoning with browser-based web exploration, enabling the agent to search, navigate web pages, and synthesize information to answer research questions.
