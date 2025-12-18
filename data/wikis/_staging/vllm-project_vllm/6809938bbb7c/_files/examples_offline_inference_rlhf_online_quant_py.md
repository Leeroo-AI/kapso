# File: `examples/offline_inference/rlhf_online_quant.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 162 |
| Classes | `MyLLM` |
| Imports | json, os, ray, rlhf_utils, torch, torchao, transformers, vllm |

## Understanding

**Status:** ✅ Explored

**Purpose:** Demonstrates RLHF with online quantization, quantizing models dynamically during training to save memory.

**Mechanism:** Extends rlhf.py pattern with torchao int8 quantization applied to models during runtime. Uses WorkerExtension to access and quantize model weights on-the-fly. Balances memory savings from quantization with training capability, applying quantization after generation but before training updates.

**Significance:** Shows advanced technique combining RLHF with dynamic quantization for maximum memory efficiency. Enables training larger models or larger batches within memory constraints by quantizing during execution.
