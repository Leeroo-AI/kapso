# File: `examples/offline_inference/rlhf_online_quant.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 162 |
| Classes | `MyLLM` |
| Imports | json, os, ray, rlhf_utils, torch, torchao, transformers, vllm |

## Understanding

**Status:** ✅ Explored

**Purpose:** Demonstrates RLHF with online quantization

**Mechanism:** Similar to rlhf.py but adds Float8DynamicActivationFloat8WeightConfig from torchao for FP8 quantization. Serializes quantization config to JSON via config_to_dict and passes via hf_overrides. Shows weight updates work with quantized inference engine.

**Significance:** Example demonstrating RLHF workflow with FP8 online quantization for memory-efficient rollout.
