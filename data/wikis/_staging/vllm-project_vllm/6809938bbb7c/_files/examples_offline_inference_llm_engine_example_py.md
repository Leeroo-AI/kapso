# File: `examples/offline_inference/llm_engine_example.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 74 |
| Functions | `create_test_prompts`, `process_requests`, `initialize_engine`, `parse_args`, `main` |
| Imports | argparse, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Low-level example of using LLMEngine directly for fine-grained control over inference execution.

**Mechanism:** Demonstrates LLMEngine API (lower-level than LLM class) with manual request management using add_request() and step() methods. Shows explicit control over batch execution, request IDs, and output collection. Processes requests synchronously with step-by-step execution control.

**Significance:** Shows how to use vLLM's engine-level API for advanced use cases requiring manual scheduling control. Important for building custom inference servers or integrations needing precise control over batch execution.
