# File: `src/peft/tuners/lora/tp_layer.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 350 |
| Classes | `LoraParallelLinear` |
| Functions | `dispatch_megatron` |
| Imports | __future__, importlib, layer, math, peft, torch, typing, warnings |

## Understanding

**Status:** ✅ Documented

**Purpose:** LoRA support for Megatron-LM tensor-parallel layers

**Mechanism:** LoraParallelLinear implements LoRA for RowParallelLinear and ColumnParallelLinear from Megatron's tensor parallelism framework. Splits lora_A for row-parallel and lora_B for column-parallel to maintain input/output shape consistency across devices. Uses forced float32 precision for LoRA parameters to avoid overflow in distributed training.

**Significance:** Critical for scaling LoRA to multi-GPU training with Megatron-LM's model parallelism. Enables training extremely large LoRA-adapted models (100B+ parameters) by properly sharding LoRA matrices across GPUs while maintaining mathematical correctness and communication efficiency. Essential for production-scale LLM fine-tuning infrastructure.
