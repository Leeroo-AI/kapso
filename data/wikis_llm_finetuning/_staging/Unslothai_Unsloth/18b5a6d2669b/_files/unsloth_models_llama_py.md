# File: `unsloth/models/llama.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 3452 |
| Classes | `LlamaRotaryEmbedding`, `LlamaLinearScalingRotaryEmbedding`, `LlamaExtendedRotaryEmbedding`, `LongRopeRotaryEmbedding`, `FastLlamaModel` |
| Functions | `original_apply_qkv`, `original_apply_o`, `fix_prepare_inputs_for_generation`, `LlamaAttention_fast_forward_inference`, `fast_swiglu_inference`, `fast_rms_layernorm_inference`, `fast_rms_layernorm_inference_gemma`, `fast_layernorm_compiled`, `... +6 more` |
| Imports | _utils, device_type, functools, gc, inspect, kernels, math, os, peft, re, ... +12 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements optimized LLaMA model support with fast attention, RoPE embeddings, SwiGLU activations, RMS LayerNorm, and efficient inference/training paths. Serves as the base architecture implementation that other models (Mistral, Qwen2, etc.) inherit from or adapt.

**Mechanism:** Provides FastLlamaModel class with pre_patch() and from_pretrained() methods that monkey-patch transformers' LlamaAttention, LlamaDecoderLayer, and LlamaForCausalLM with optimized implementations. Key optimizations include: fast attention dispatch (Flash Attention, xformers, SDPA), custom RoPE embeddings with dynamic extension, fused SwiGLU/RMS LayerNorm kernels, KV cache management, and specialized inference paths. Uses attention_dispatch module for backend selection.

**Significance:** Foundation architecture for Unsloth's model zoo. Most other model implementations (Mistral, Qwen, Gemma) inherit from FastLlamaModel or reuse its optimized components. Contains the core attention and MLP optimization strategies that enable 2x+ speedups. At 3452 lines, this is the largest and most complete model implementation in the codebase.
