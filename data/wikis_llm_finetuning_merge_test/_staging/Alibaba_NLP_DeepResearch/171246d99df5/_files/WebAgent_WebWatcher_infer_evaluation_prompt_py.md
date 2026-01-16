# File: `WebAgent/WebWatcher/infer/evaluation/prompt.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 458 |

## Understanding

**Status:** ✅ Explored

**Purpose:** Central repository of prompt templates used throughout the WebWatcher evaluation system for system instructions, content extraction, and answer judging across multiple benchmarks.

**Mechanism:** Contains a comprehensive collection of string constants defining various prompts:
1. **Agent System Prompts:**
   - `SYSTEM_PROMPT_MULTI`: Defines agent persona as "Web Information Seeking Master" with principles for persistent actions, repeated verification, and attention to detail
   - `USER_PROMPT`: Tool-use format template with `search` and `visit` tools, showing think/tool_call/tool_response/answer cycle
2. **Content Extraction:**
   - `EXTRACTOR_PROMPT`: Template for processing webpage content against user goals, extracting evidence with rational/evidence/summary JSON output
3. **Judge Prompts for Different Benchmarks:**
   - `JUDGE_PROMPT_GAIA`: Simple equivalence checking for GAIA benchmark
   - `JUDGE_PROMPT_BC_zh/en`: Detailed Chinese/English prompts for BrowseComp benchmark with extensive examples of correct/incorrect judgments
   - `JUDGE_PROMPT_QA`: Three-category grading (CORRECT/INCORRECT/NOT_ATTEMPTED) with comprehensive edge case handling
   - `JUDGE_PROMPT_CONFIDENCE`: Extracts confidence scores alongside correctness judgments
   - `JUDGE_PROMPT_SEAL0_QA`: Variant for SEAL benchmark evaluation
   - `JUDGE_PROMPT_XBENCH`: Chinese-language evaluation prompt for XBench benchmark
   - `JUDGE_PROMPT_BROWSECOMP_OFFICIAL`: Official BrowseComp evaluation format

**Significance:** Foundational configuration file that standardizes evaluation criteria across diverse benchmarks. The carefully crafted prompts ensure consistent, fair, and reproducible evaluation of web agent responses, handling nuances like numerical tolerance, name translations, partial answers, and language variations. Critical for maintaining evaluation quality and benchmark comparability.
