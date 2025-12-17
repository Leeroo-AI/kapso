# File: `examples/offline_inference/lora_with_quantization_inference.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 127 |
| Functions | `create_test_prompts`, `process_requests`, `initialize_engine`, `main` |
| Imports | gc, huggingface_hub, torch, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Demonstrates LoRA with quantization (QLoRA, AWQ+LoRA, GPTQ+LoRA)

**Mechanism:** Tests three quantization methods combined with LoRA: bitsandbytes (QLoRA), AWQ, and GPTQ. For each config, initializes LLMEngine with enable_lora, max_lora_rank, max_loras, runs prompts both with and without LoRA requests, then cleans up GPU memory between tests.

**Significance:** Example showing LoRA inference works with various quantization techniques for memory-efficient fine-tuned model serving.
