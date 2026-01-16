# File: `WebAgent/WebResummer/src/judge_prompt.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 150 |

## Understanding

**Status:** ✅ Explored

**Purpose:** Defines LLM judge prompts for evaluating whether predicted answers match ground-truth answers across different benchmarks and languages.

**Mechanism:** Contains three prompt templates: `JUDGE_PROMPT_GAIA` for the GAIA benchmark (simple English format asking for "Correct"/"Incorrect"), `JUDGE_PROMPT_BC_zh` for BrowseComp Chinese benchmark, and `JUDGE_PROMPT_BC_en` for BrowseComp English benchmark. The BrowseComp prompts are comprehensive with detailed examples of correct/incorrect classifications, handling nuances like partial matches, numerical tolerances, translation variations, multi-aspect answers (using bracket notation), and common edge cases. Prompts use placeholder variables ({question}, {correct_answer}, {response}) for formatting.

**Significance:** Essential utility for the evaluation pipeline. Provides carefully crafted prompts that enable accurate LLM-based answer verification, supporting multilingual evaluation with detailed classification guidelines to ensure consistent and fair judging across different answer formats.
