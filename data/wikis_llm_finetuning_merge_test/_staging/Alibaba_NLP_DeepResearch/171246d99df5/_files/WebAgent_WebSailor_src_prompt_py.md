# File: `WebAgent/WebSailor/src/prompt.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 206 |

## Understanding

**Status:** ✅ Explored

**Purpose:** Defines all prompt templates used by the WebSailor agent system, including system prompts for the agent, user prompts with tool definitions, extraction prompts for webpage content processing, and judge prompts for evaluation.

**Mechanism:** Contains four main prompt categories: (1) `SYSTEM_PROMPT_MULTI` - establishes the agent as a "Web Information Seeking Master" with principles for persistent exploration and verification; (2) `EXTRACTOR_PROMPT` - instructs content extraction from webpages with rational, evidence, and summary fields in JSON format; (3) `USER_PROMPT` - defines the tool interface (search and visit tools) with ReAct-style think/tool_call/tool_response/answer format; (4) Judge prompts (`JUDGE_PROMPT_GAIA`, `JUDGE_PROMPT_BC`, `JUDGE_PROMPT_QA`) for evaluating answer correctness with dataset-specific criteria.

**Significance:** Central configuration file that defines the agent's behavior, tool interfaces, and evaluation criteria. The prompts shape how the agent reasons about web information seeking tasks and how its outputs are evaluated.
