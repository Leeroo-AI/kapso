# File: `unsloth/models/llama.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 3400 |
| Classes | `LlamaRotaryEmbedding`, `LlamaLinearScalingRotaryEmbedding`, `LlamaExtendedRotaryEmbedding`, `LongRopeRotaryEmbedding`, `FastLlamaModel` |
| Functions | `original_apply_qkv`, `original_apply_o`, `fix_prepare_inputs_for_generation`, `LlamaAttention_fast_forward_inference`, `fast_swiglu_inference`, `fast_rms_layernorm_inference`, `fast_rms_layernorm_inference_gemma`, `fast_layernorm_compiled`, `... +6 more` |
| Imports | _utils, device_type, functools, gc, inspect, kernels, math, os, peft, re, ... +12 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Reference architecture implementation for Llama models with attention/rope optimization and inference kernels.

**Mechanism:** Patches LlamaAttention with fast_forward kernels using flash attention and SDPA, implements custom RoPE embeddings with dynamic extension, provides FastLlamaModel base class with from_pretrained/patch_peft_model methods, includes inference-specific paging attention.

**Significance:** Foundation for all Llama-based models providing reusable attention kernels and optimization patterns. Most other model implementations inherit or adapt from this.
