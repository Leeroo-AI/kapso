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

**Purpose:** Reference implementation of optimized Llama model architecture with performance enhancements

**Mechanism:** Implements `FastLlamaModel` class with heavily optimized forward passes for attention, MLP, and RMSNorm layers. Key optimizations include:
- Custom fast inference paths using KV cache with paged attention
- Optimized RoPE (Rotary Position Embeddings) implementations
- Fast SwiGLU activation with pre-allocated buffers
- Custom attention mechanisms (SDPA, Flash Attention, xformers)
- Memory-efficient gradient checkpointing
- Pre-allocated tensor buffers to avoid allocations during inference
- Batched operations and fused kernels

The file also provides patching functions to replace transformers' default implementations with Unsloth's optimized versions, handling both training and inference modes.

**Significance:** This is the foundational model implementation that other architecture-specific files (Mistral, Gemma, Qwen, etc.) inherit from and extend. It establishes the optimization patterns used throughout Unsloth:
- Pre-computation and caching strategies
- Efficient memory management
- Kernel fusion opportunities
- Backward compatibility with HuggingFace transformers
- Support for various attention backends
- PEFT/LoRA integration for parameter-efficient fine-tuning

Most other model files follow this structure, customizing only architecture-specific differences like attention mechanisms, MLP structures, or normalization layers.
