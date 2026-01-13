# File: `unsloth/models/llama.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 3475 |
| Classes | `LlamaRotaryEmbedding`, `LlamaLinearScalingRotaryEmbedding`, `LlamaExtendedRotaryEmbedding`, `LongRopeRotaryEmbedding`, `FastLlamaModel` |
| Functions | `original_apply_qkv`, `original_apply_o`, `fix_prepare_inputs_for_generation`, `LlamaAttention_fast_forward_inference`, `fast_swiglu_inference`, `fast_rms_layernorm_inference`, `fast_rms_layernorm_inference_gemma`, `fast_layernorm_compiled`, `... +6 more` |
| Imports | _utils, device_type, functools, gc, inspect, kernels, math, os, peft, re, ... +12 more |

## Understanding

**Status:** Explored

**Purpose:** Core FastLlamaModel implementation providing optimized Llama model loading, PEFT/LoRA integration, custom RoPE embeddings, and fast inference/training forward passes.

**Mechanism:** `FastLlamaModel` class implements `from_pretrained()` for model loading with automatic tokenizer fixing, `get_peft_model()` for LoRA adapter setup with target module detection, `for_inference()` and `for_training()` mode switching. Defines custom RoPE classes: `LlamaRotaryEmbedding`, `LlamaLinearScalingRotaryEmbedding`, `LlamaExtendedRotaryEmbedding`, `LongRopeRotaryEmbedding` for various context length extension methods. Implements optimized forwards: `LlamaAttention_fast_forward` using Flash Attention/xformers, `LlamaDecoderLayer_fast_forward`, `LlamaModel_fast_forward`. Provides KV cache optimizations in `LlamaAttention_fast_forward_inference`. Includes LoRA application functions: `apply_lora_qkv`, `apply_lora_o`, `apply_lora_mlp_swiglu` for fused attention operations.

**Significance:** Core component - the foundational model implementation that other FastXModel classes inherit from or reference. Contains the primary optimizations that give Unsloth its speed advantages: fused attention, optimized RoPE, and efficient LoRA application.
