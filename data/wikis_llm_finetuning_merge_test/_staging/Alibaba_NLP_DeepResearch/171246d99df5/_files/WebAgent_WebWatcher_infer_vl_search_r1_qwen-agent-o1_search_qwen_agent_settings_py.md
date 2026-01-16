# File: `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/settings.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 27 |
| Imports | ast, os, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Centralizes configuration constants for the qwen_agent package, covering LLM settings, agent behavior limits, tool configuration, and RAG (Retrieval-Augmented Generation) parameters.

**Mechanism:** Defines environment-variable-backed configuration constants: (1) LLM settings: `DEFAULT_MAX_INPUT_TOKENS` (30000) for message truncation, (2) Agent settings: `MAX_LLM_CALL_PER_RUN` (8) limits LLM calls per agent run, (3) Tool settings: `DEFAULT_WORKSPACE` for tool file storage, (4) RAG settings: `DEFAULT_MAX_REF_TOKEN` (20000) for reference material window, `DEFAULT_PARSER_PAGE_SIZE` (500) for chunk size, `DEFAULT_RAG_KEYGEN_STRATEGY` for keyword generation, `DEFAULT_RAG_SEARCHERS` for hybrid retrieval methods, (5) VL settings: `DEFAULT_ENABLE_SWITCH_VL` for vision-language switching. All values can be overridden via environment variables (e.g., `QWEN_AGENT_DEFAULT_MAX_INPUT_TOKENS`).

**Significance:** This is a core configuration component that allows runtime customization of agent behavior without code changes. It provides sensible defaults while enabling deployment-specific tuning through environment variables, which is essential for different use cases (e.g., longer context for complex queries, different RAG strategies).
