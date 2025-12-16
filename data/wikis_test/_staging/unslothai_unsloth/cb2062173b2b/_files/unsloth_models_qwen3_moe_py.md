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

**Purpose:** Extends Qwen3 support to Mixture-of-Experts (MoE) architectures by optimizing sparse expert routing and computation.

**Mechanism:**
- Extends `FastQwen3Model` and reuses its attention optimizations via `Qwen3Attention_fast_forward`
- **MoE-specific optimization in `Qwen3MoeSparseMoeBlock_fast_forward`**:
  - Uses `fast_linear_forward` for the gate projection that determines expert routing
  - Implements top-k expert selection with normalized routing weights
  - Processes each expert sequentially, computing expert outputs weighted by routing probabilities
  - Uses `torch.index_add_` for efficient accumulation of expert outputs
- **Decoder layer handling**: Distinguishes between inference mode (with `fast_rms_layernorm_inference`) and training mode for proper normalization
- Each expert uses `fast_swiglu_inference` (line 193) which is the same optimization as dense models' MLP layers
- Maintains generation flag (`_flag_for_generation`) to switch between fast inference and training paths
- Patches all attention variants (`Qwen3MoeAttention`, potentially SDPA/FlashAttention2) with the same forward implementation

**Significance:** Enables efficient fine-tuning and inference of Qwen3's MoE variants, which are computationally intensive due to multiple expert networks. The optimization is critical because MoE models can have significantly more parameters while maintaining computational efficiency through sparse activation. Shows Unsloth's capability to handle advanced architectures beyond standard dense transformers, supporting the growing trend of MoE models for improved performance-per-parameter ratios.
