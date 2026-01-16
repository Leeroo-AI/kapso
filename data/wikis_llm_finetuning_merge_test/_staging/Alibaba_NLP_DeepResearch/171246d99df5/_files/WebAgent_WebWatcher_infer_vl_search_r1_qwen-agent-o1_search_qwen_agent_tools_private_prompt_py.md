# File: `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/tools/private/prompt.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 206 |

## Understanding

**Status:** ✅ Explored

**Purpose:** Defines all system prompts, user prompts, and evaluation judge prompts used throughout the web agent reasoning and evaluation pipeline.

**Mechanism:** Contains several key prompt templates:
- `SYSTEM_PROMPT_MULTI`: Instructs the model to be a "Web Information Seeking Master" with persistent exploration, repeated verification, and attention to detail
- `EXTRACTOR_PROMPT`: Template for extracting rational/evidence/summary from webpage content given a user goal
- `USER_PROMPT`: Defines the tool-use protocol with search and visit tools, showing the think/tool_call/tool_response/answer format
- `JUDGE_PROMPT_GAIA`: Simple equivalence judge for GAIA benchmark
- `JUDGE_PROMPT_BC`: Detailed judge prompt with extracted_final_answer, reasoning, correct, and confidence fields
- `JUDGE_PROMPT_QA`: Comprehensive grading rubric (CORRECT/INCORRECT/NOT_ATTEMPTED) with extensive examples for semantic matching

**Significance:** Central configuration file for the agent's behavior and evaluation. The prompts shape how the agent approaches information seeking (persistent, verified), how it processes webpage content (structured extraction), and how responses are evaluated. Critical for reproducibility and consistent agent behavior across experiments.
