# File: `evaluation/prompt.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 458 |

## Understanding

**Status:** ✅ Explored

**Purpose:** Centralizes all prompt templates used by the DeepResearch system for both agent operation and evaluation judging across multiple benchmarks and languages.

**Mechanism:** Contains three categories of prompts: (1) **System/Agent prompts** - `SYSTEM_PROMPT_MULTI` defines the web information seeking agent persona emphasizing persistence and verification; `USER_PROMPT` provides tool definitions (search, visit) and the structured interaction format using XML-style tags (<think>, <tool_call>, <tool_response>, <answer>); `EXTRACTOR_PROMPT` guides webpage content extraction with rational/evidence/summary structure. (2) **Judge prompts for evaluation** - Multiple language and benchmark-specific prompts: `JUDGE_PROMPT_GAIA` for simple equivalence checking, `JUDGE_PROMPT_BC_zh`/`JUDGE_PROMPT_BC_en` for BrowseComp (Chinese/English) with detailed examples of correct/incorrect classifications, `JUDGE_PROMPT_QA` and `JUDGE_PROMPT_SEAL0_QA` for general QA with three-way grading (CORRECT/INCORRECT/NOT_ATTEMPTED), `JUDGE_PROMPT_CONFIDENCE` and `JUDGE_PROMPT_BROWSECOMP_OFFICIAL` for structured JSON output with confidence extraction, `JUDGE_PROMPT_XBENCH` for Chinese XBench evaluation.

**Significance:** Critical configuration file that defines the behavioral contract between the agent and its tools, as well as the evaluation criteria for judging model outputs. The detailed judge prompts with extensive examples ensure consistent and fair evaluation across different benchmarks, handling edge cases like partial matches, numerical tolerances, name transliterations, and multi-part answers.
