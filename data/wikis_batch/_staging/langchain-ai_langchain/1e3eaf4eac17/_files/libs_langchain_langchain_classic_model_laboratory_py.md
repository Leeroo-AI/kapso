# File: `libs/langchain/langchain_classic/model_laboratory.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 98 |
| Classes | `ModelLaboratory` |
| Imports | __future__, collections, langchain_classic, langchain_core |

## Understanding

**Status:** ✅ Explored

**Purpose:** Interactive tool for comparing LLM model outputs side-by-side

**Mechanism:** The `ModelLaboratory` class wraps multiple LLM chains and runs them in parallel on the same input, displaying outputs in colored text for easy comparison. Supports initialization from either Chain objects or LLM instances with a shared prompt template.

**Significance:** Development and evaluation utility that helps developers experiment with and compare different language models or configurations to determine which performs best for their use case. Validates that all chains have exactly one input and one output variable for consistent comparison.
