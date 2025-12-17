# File: `examples/offline_inference/multilora_inference.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 106 |
| Functions | `create_test_prompts`, `process_requests`, `initialize_engine`, `main` |
| Imports | huggingface_hub, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Demonstrates multiple LoRA adapters with batching

**Mechanism:** Shows 2 base model requests and 4 LoRA requests using 2 different LoRA adapters (sql-lora, sql-lora2). With max_loras=1, demonstrates sequential execution where second adapter waits for first to complete. Uses LLMEngine with enable_lora, max_lora_rank, max_cpu_loras for LoRA memory management.

**Significance:** Example demonstrating multi-LoRA request handling with adapter swapping and CPU cache management.
