# File: `unsloth/models/gemma2.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 656 |
| Classes | `FastGemma2Model` |
| Functions | `Gemma2Attention_fast_forward`, `Gemma2DecoderLayer_fast_forward`, `Gemma2Attention_fast_forward_inference`, `Gemma2Model_fast_forward_inference` |
| Imports | _utils, gemma, llama, math, transformers, unsloth_zoo, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Optimized implementation of Gemma 2 architecture with logit softcapping support, sliding window attention, and both local and global attention patterns specific to Gemma 2's dual-attention design.

**Mechanism:** Implements FastGemma2Model extending FastGemmaModel with Gemma 2-specific features. The key distinction is Gemma2Attention_fast_forward which handles attention softcapping (clamping logits to prevent overflow), alternating sliding window and full attention layers, and both training and inference paths. Supports flash-attention 2.6.3+ for hardware-accelerated softcapping when available, falling back to SDPA with manual softcapping otherwise. Uses GemmaFixedRotaryEmbedding from parent gemma module and inherits fast_geglu_inference. The decoder layer forward handles the interleaved attention pattern (every other layer uses sliding window). Inference mode includes specialized optimizations with memory reuse and fused operations.

**Significance:** Enables efficient training and inference for Gemma 2 models (2B, 9B, 27B) which introduced architectural improvements over Gemma 1 including attention softcapping for stability and sliding windows for efficiency. The softcapping support is critical as it's a key innovation in Gemma 2. The alternating attention pattern optimization reduces memory usage while maintaining model quality. The conditional flash-attention support ensures users with modern GPUs get maximum performance while maintaining compatibility with older hardware.
