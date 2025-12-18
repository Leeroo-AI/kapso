# File: `src/transformers/masking_utils.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 1381 |
| Classes | `AttentionMaskInterface`, `AttentionMask` |
| Functions | `and_masks`, `or_masks`, `causal_mask_function`, `bidirectional_mask_function`, `sliding_window_overlay`, `chunked_overlay`, `sliding_window_causal_mask_function`, `chunked_causal_mask_function`, `... +16 more` |
| Imports | cache_utils, collections, configuration_utils, itertools, torch, typing, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Modern unified attention mask creation system supporting causal, sliding window, chunked, and bidirectional patterns across different attention implementations (SDPA, eager, FlashAttention, FlexAttention).

**Mechanism:** Provides composable mask functions (causal_mask_function, sliding_window_overlay, chunked_overlay, padding_mask_function) that can be combined with and_masks/or_masks operators. Main interfaces (sdpa_mask, eager_mask, flash_attention_mask, flex_attention_mask) create implementation-specific masks from these primitives. Includes smart optimizations: skipping mask creation when possible (e.g., using SDPA's is_causal argument), handling packed sequence format detection via position_ids, and supporting custom mask overlays through or_mask_function/and_mask_function parameters. The AttentionMaskInterface provides a registry for attention implementations. High-level functions (create_causal_mask, create_sliding_window_causal_mask, create_chunked_causal_mask, create_bidirectional_mask) offer model-friendly APIs.

**Significance:** This is the modern replacement for modeling_attn_mask_utils, providing a much more flexible and efficient attention masking system. It supports diverse attention patterns (Mistral's sliding window, Llama4's chunked attention), works across all attention backends, and enables compile-friendly optimizations. Essential for performance and correctness of transformer attention mechanisms.
