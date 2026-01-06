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

**Purpose:** Demonstrates combining LoRA adapters with quantized base models for memory-efficient fine-tuned inference.

**Mechanism:** Loads a quantized base model (GPTQ or AWQ) with enable_lora=True and applies LoRA adapters on top. Shows how to use LoRARequest to switch between different adapters during inference, enabling multiple task-specific models from a single quantized base. Uses LLMEngine for low-level control.

**Significance:** Illustrates powerful technique combining quantization (for base model compression) with LoRA (for task adaptation), maximizing GPU memory efficiency while supporting multiple specialized models. Critical for production scenarios with limited GPU memory.
