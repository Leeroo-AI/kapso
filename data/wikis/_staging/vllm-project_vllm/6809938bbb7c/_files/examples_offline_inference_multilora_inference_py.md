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

**Purpose:** Demonstrates serving multiple LoRA adapters simultaneously from a single base model using LLMEngine.

**Mechanism:** Loads SQL-focused LoRA adapters and processes different requests with different LoRA adapters via LoRARequest objects. Shows how to batch requests using different adapters together, with vLLM efficiently swapping adapter weights during execution. Uses enable_lora=True and max_loras configuration.

**Significance:** Critical pattern for multi-tenant or multi-task inference where different users/tasks need different fine-tuned models. Shows how vLLM enables efficient adapter switching without loading separate model instances.
