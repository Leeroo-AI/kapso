# File: `unsloth/models/qwen3_moe.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 243 |
| Classes | `FastQwen3MoeModel` |
| Functions | `Qwen3MoeSparseMoeBlock_fast_forward`, `Qwen3MoeDecoderLayer_fast_forward` |
| Imports | _utils, llama, os, qwen3, transformers, unsloth_zoo |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides optimized implementations for Alibaba's Qwen3 Mixture of Experts (MoE) model architecture, combining Qwen3's QK normalization with sparse expert routing.

**Mechanism:** Implements `FastQwen3MoeModel` extending `FastQwen3Model` with: (1) `Qwen3MoeSparseMoeBlock_fast_forward` implementing top-k expert routing - applies softmax to router logits, selects top-k experts per token, computes weighted expert outputs, and uses `index_add_` for efficient accumulation; (2) `Qwen3MoeDecoderLayer_fast_forward` integrating the MoE block as the MLP layer with proper router logits output handling; (3) Reuses `Qwen3Attention_fast_forward` from qwen3.py for QK-normalized attention. The MoE block iterates over `self.num_experts`, applying each expert's MLP (`Qwen3MoeMLP.forward = fast_swiglu_inference`) to tokens routed to that expert, weighted by routing weights. Expert mask is computed via one-hot encoding of selected experts.

**Significance:** Core model architecture support for Qwen3-MoE sparse models. MoE architectures achieve better compute efficiency by activating only a subset of parameters per token. This file combines Qwen3's QK normalization with efficient expert routing and Unsloth's fast SwiGLU inference, enabling memory-efficient fine-tuning of large MoE models.
