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

**Purpose:** Implements optimized Qwen3 Mixture-of-Experts (MoE) model support, combining Qwen3's QK-norm attention with sparse expert routing where only top-k experts process each token.

**Mechanism:** Inherits Qwen3Attention_fast_forward from qwen3.py for attention with QK-norm. Implements custom Qwen3MoeSparseMoeBlock_fast_forward that: 1) routes tokens to top_k experts via softmax(gate_proj(X)), 2) processes each expert's assigned tokens through fast_swiglu_inference (reused from dense models), 3) aggregates expert outputs weighted by routing probabilities using torch.Tensor.index_add_. The MoE block replaces standard MLP layers in decoder. Patches Qwen3MoeMLP.forward = fast_swiglu_inference since expert MLPs are standard SwiGLU.

**Significance:** Enables efficient fine-tuning of Qwen3 MoE models (e.g., Qwen3-30B-A3B with 30B params, 3B active). MoE architectures are becoming standard for efficient large models. The expert routing optimization is crucial - naive implementations would process all experts. At 243 lines, it's compact due to reusing both Qwen3 attention and LLaMA MLP optimizations.
