# File: `evaluation/evaluate_hle_official.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 243 |
| Classes | `ExtractedAnswer` |
| Functions | `load_jsonl`, `write_jsonl`, `get_client`, `extract_answer`, `extract_response`, `process_item` |
| Imports | argparse, collections, concurrent, json, openai, os, pydantic, re, threading, time, ... +3 more |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Evaluates model predictions on HLE (Humanity's Last Exam) benchmark using LLM-as-judge methodology with structured output parsing and comprehensive metrics reporting.

**Mechanism:** The script loads predictions from a JSONL file and processes each item through an LLM judge (o3-mini by default). Using Pydantic's `ExtractedAnswer` model with structured output (`response_format`), it extracts: the model's final answer, reasoning, correctness judgment (yes/no), and confidence score. Processing is parallelized via ThreadPoolExecutor (20 workers). The script computes aggregate metrics including: accuracy rate, answer extraction success rate, token usage (prompt/completion), tool usage statistics per question type (google_search, google_scholar, Visit, PythonInterpreter), average turns per question, and estimated API costs for different model pricing tiers.

**Significance:** Official evaluation script for HLE benchmark, which tests AI systems on expert-level questions across diverse domains. The script provides detailed cost analysis and tool usage breakdowns essential for understanding the computational requirements and search strategies of deep research agents on challenging questions.
