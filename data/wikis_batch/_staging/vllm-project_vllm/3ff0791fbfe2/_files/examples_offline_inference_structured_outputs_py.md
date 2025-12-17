# File: `examples/offline_inference/structured_outputs.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 113 |
| Classes | `CarType`, `CarDescription` |
| Functions | `format_output`, `generate_output`, `main` |
| Imports | enum, pydantic, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Demonstrates structured output constraints

**Mechanism:** Shows four structured output modes: Choice (list of options), Regex patterns, JSON schema via Pydantic models, and Grammar (EBNF-style). Uses StructuredOutputsParams with different constraint types to force model outputs into specific formats.

**Significance:** Example demonstrating vLLM's structured output capabilities for constrained generation.
