# src/transformers/masking_utils.py

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides a modern, unified attention masking framework supporting multiple attention backends (SDPA, eager, Flash Attention, flex attention) with flexible mask patterns (causal, bidirectional, sliding window, chunked) for transformer models.

**Mechanism:** The file implements a comprehensive masking system:
- **Mask factory functions**: Pure functions like `causal_mask_function()`, `sliding_window_overlay()`, `chunked_overlay()` that define mask patterns
- **Mask combinators**: `and_masks()` and `or_masks()` for composing complex mask patterns
- **Backend-specific mask creation**: Separate functions for each attention implementation:
  - `sdpa_mask()`: Boolean masks for PyTorch SDPA
  - `eager_mask()`: Float masks with -inf for eager attention
  - `flash_attention_mask()`: Optimized 2D masks for Flash Attention
  - `flex_attention_mask()`: BlockMask for flex attention
- **AttentionMaskInterface**: Unified interface mapping attention implementations to mask functions
- **High-level mask creators**: `create_causal_mask()`, `create_sliding_window_causal_mask()`, `create_chunked_causal_mask()`, `create_bidirectional_mask()`
- **Optimization**: Detects when masks can be skipped (e.g., single query token, full attention) to use backend optimizations
- **Packed sequence support**: `find_packed_sequence_indices()` for handling multiple sequences in a single batch dimension

**Significance:** This module replaces the deprecated `modeling_attn_mask_utils.py` with a cleaner, more extensible architecture. It enables models to seamlessly switch between attention backends while maintaining correct mask semantics, supports advanced attention patterns like sliding windows (Mistral) and chunked attention (Llama4), and provides critical performance optimizations by avoiding unnecessary mask materialization. The unified interface simplifies model implementation and ensures consistent attention behavior across different hardware and software configurations.
