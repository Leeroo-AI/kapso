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

**Purpose:** Core implementation of optimized Llama model architecture with fast attention mechanisms, efficient inference kernels, custom RoPE embeddings, and gradient checkpointing support.

**Mechanism:** Implements FastLlamaModel class with optimized forward passes for training and inference. Provides custom RoPE embedding classes (standard, linear scaling, extended, long rope) that support dynamic rope extension and caching. Implements fast forward functions for attention (LlamaAttention_fast_forward) using custom kernels from utils.attention_dispatch, decoder layers with optimized SwiGLU and RMSNorm, and model-level forwards with gradient checkpointing. Includes fast inference paths that fuse operations and reduce memory allocations. The from_pretrained method handles model loading with quantization, tokenizer patching, LoRA adapter support, and model patching. Provides patch_peft_model for post-load adapter optimization.

**Significance:** Foundational model implementation that serves as the base for many other architectures. The optimizations in this file (fused kernels, efficient attention dispatch, smart gradient checkpointing) are critical for Unsloth's 2x faster training and 2x memory reduction claims. The Llama architecture patterns and optimizations are inherited by Mistral, Qwen, and other derivative architectures. The comprehensive RoPE embedding support enables handling various context length scaling strategies.
