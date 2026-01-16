# File: `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/agent.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 316 |
| Classes | `Agent`, `BasicAgent` |
| Imports | abc, copy, json, qwen_agent, traceback, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Defines the core `Agent` abstract base class and `BasicAgent` implementation that form the foundation for all agent-based workflows in the Qwen-Agent framework.

**Mechanism:** The `Agent` class provides: (1) LLM integration via `_call_llm()` which interfaces with chat models, (2) Tool management through `_init_tool()` and `_call_tool()` methods that register and invoke tools from a registry, (3) Message handling with automatic system message injection and language detection (Chinese/English), (4) Both streaming (`run()`) and non-streaming (`run_nonstream()`) response generation, (5) Batch processing via `run_batch()` with parallel execution (up to 20 workers). Subclasses must implement the abstract `_run()` method to define their workflow. `BasicAgent` is a minimal implementation that simply forwards messages to the LLM.

**Significance:** This is the most critical core component in the qwen_agent package. It establishes the agent architecture pattern used throughout the system, enabling LLM-powered agents that can use tools, process messages, and generate responses. All specialized agents (search agents, RAG agents, etc.) inherit from this base class.
