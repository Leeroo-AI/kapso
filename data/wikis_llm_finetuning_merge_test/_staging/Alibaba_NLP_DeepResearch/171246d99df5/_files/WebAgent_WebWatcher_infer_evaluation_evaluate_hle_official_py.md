# File: `WebAgent/WebWatcher/infer/evaluation/evaluate_hle_official.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 245 |
| Classes | `ExtractedAnswer` |
| Functions | `load_jsonl`, `write_jsonl`, `get_client`, `extract_answer`, `extract_response`, `process_item` |
| Imports | argparse, collections, concurrent, json, openai, os, pydantic, re, threading, time, ... +3 more |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Evaluates model responses against ground truth answers for the HLE (Humanity's Last Exam) benchmark using an LLM judge to determine correctness.

**Mechanism:** The file implements an evaluation pipeline that:
1. Uses `ExtractedAnswer` Pydantic model for structured response parsing with fields: extracted_final_answer, reasoning, correct, confidence, strict
2. `extract_answer()` sends questions, model responses, and correct answers to a judge model (default: Qwen2.5-32B-Instruct) via OpenAI-compatible API
3. `extract_response()` parses model responses to extract final answers from `<answer>` tags
4. `process_item()` orchestrates the evaluation of individual items, computing accuracy metrics
5. Main script processes JSONL input files using ThreadPoolExecutor for parallel evaluation
6. Generates comprehensive reports including accuracy, token usage, tool usage statistics, and cost estimates (for o4-mini and Claude)
7. Outputs detailed evaluation results to `.eval_details.jsonl` and summary report to `.report.json`

**Significance:** Core evaluation component for assessing WebWatcher agent performance on challenging benchmark questions. Provides standardized metrics for comparing model capabilities on the HLE dataset, which tests model knowledge boundaries. Essential for validating research outcomes and measuring improvements in web information-seeking agents.
