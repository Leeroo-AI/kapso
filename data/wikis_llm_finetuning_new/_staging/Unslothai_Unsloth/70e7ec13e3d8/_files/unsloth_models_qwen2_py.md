# File: `unsloth/models/qwen2.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 101 |
| Classes | `FastQwen2Model` |
| Imports | llama, transformers |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides optimized implementations for Alibaba's Qwen2 model architecture, which is architecturally similar to Llama with minimal customization needed.

**Mechanism:** Implements `FastQwen2Model` extending `FastLlamaModel` with a minimal adapter approach: (1) `pre_patch` method patches `Qwen2Attention`, `Qwen2SdpaAttention`, and `Qwen2FlashAttention2` to use `LlamaAttention_fast_forward`; (2) `Qwen2DecoderLayer` uses `LlamaDecoderLayer_fast_forward`; (3) `Qwen2Model` uses `LlamaModel_fast_forward`; (4) `Qwen2ForCausalLM` uses `CausalLM_fast_forward` with `LlamaModel_fast_forward_inference`. Uses `LlamaRotaryEmbedding` and `LlamaLinearScalingRotaryEmbedding` directly. The `from_pretrained` method delegates to `FastLlamaModel.from_pretrained` with `model_patcher=FastQwen2Model`.

**Significance:** Model architecture support for Qwen2 models. This is a lightweight adapter file (~100 lines) demonstrating that Qwen2's architecture is sufficiently similar to Llama that it can reuse nearly all Llama optimizations. The file primarily exists to properly wire up the transformers Qwen2 classes to Unsloth's fast forward methods and replace the rotary embedding implementation.
