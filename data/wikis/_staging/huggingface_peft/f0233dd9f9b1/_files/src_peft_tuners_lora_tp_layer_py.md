# File: `src/peft/tuners/lora/tp_layer.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 350 |
| Classes | `LoraParallelLinear` |
| Functions | `dispatch_megatron` |
| Imports | __future__, importlib, layer, math, peft, torch, typing, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tensor parallel LoRA layers

**Mechanism:** Implements LoRA for Megatron-LM tensor-parallel models. LoraParallelLinear handles both RowParallelLinear (splits lora_A across rows) and ColumnParallelLinear (splits lora_B across columns) to maintain parallel computation patterns. Creates parallel LoRA matrices using Megatron's parallel primitives with proper input_is_parallel and gather_output settings. Forward pass calls parallel base layer, adds parallel LoRA contribution. Supports merge/unmerge with parallel weight updates. dispatch_megatron() detects Megatron parallel layers and creates appropriate wrappers. Forces float32 for LoRA weights to prevent overflow.

**Significance:** Essential for training massive models with tensor parallelism using Megatron-LM. Enables LoRA fine-tuning on models too large for single GPU by properly distributing LoRA matrices across devices. Critical for enterprise-scale model training where multi-GPU parallelism is mandatory. Maintains computational efficiency of parallel training.
