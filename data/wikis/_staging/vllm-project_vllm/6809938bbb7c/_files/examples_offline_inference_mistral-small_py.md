# File: `examples/offline_inference/mistral-small.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 186 |
| Functions | `run_simple_demo`, `run_advanced_demo`, `parse_args`, `main` |
| Imports | argparse, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Comprehensive example for Mistral-Small 3.1 model showing both simple chat and advanced prompt continuation patterns.

**Mechanism:** Provides two usage patterns: (1) simple demo using llm.chat() with message-based API for conversational interactions, and (2) advanced demo using llm.generate() with raw prompt strings for fine-grained control over prompt formatting. Demonstrates Mistral's specific chat template and tokenization requirements.

**Significance:** Model-specific reference for Mistral-Small 3.1, showing proper usage patterns for both high-level chat API and low-level generate API. Important for understanding how to work with Mistral's instruction format and capabilities.
