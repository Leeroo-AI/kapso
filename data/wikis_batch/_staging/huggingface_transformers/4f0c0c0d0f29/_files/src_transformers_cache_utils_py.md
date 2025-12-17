# File: `src/transformers/cache_utils.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 1402 |
| Classes | `CacheLayerMixin`, `DynamicLayer`, `DynamicSlidingWindowLayer`, `StaticLayer`, `StaticSlidingWindowLayer`, `QuantizedLayer`, `QuantoQuantizedLayer`, `HQQQuantizedLayer`, `Cache`, `DynamicCache`, `StaticCache`, `QuantizedCache`, `EncoderDecoderCache`, `SlidingWindowLayer`, `ChunkedSlidingLayer`, `OffloadedCache`, `OffloadedStaticCache`, `SlidingWindowCache`, `HybridCache`, `HybridChunkedCache`, `OffloadedHybridCache`, `QuantoQuantizedCache`, `HQQQuantizedCache` |
| Imports | abc, collections, configuration_utils, torch, typing, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Manages key-value cache storage for transformer attention mechanisms, providing multiple strategies for memory efficiency and generation speed optimization.

**Mechanism:** Implements hierarchical cache system with CacheLayerMixin base and specialized variants: DynamicCache (grows with generation), StaticCache (preallocated for torch.compile), SlidingWindowCache (fixed window size), QuantizedCache (4-bit/2-bit quantization via Quanto/HQQ), and EncoderDecoderCache (separate self/cross-attention caches). Supports layer offloading to CPU with prefetching, batch operations (reorder, crop, repeat), and hybrid configurations based on model config layer_types. StaticCache uses torch._dynamo.mark_static_address for compilation compatibility.

**Significance:** Critical for efficient text generation, enabling: (1) fast inference by avoiding recomputation of past attention, (2) long sequence generation through quantization/offloading, (3) torch.compile support via static caches, and (4) memory optimization for resource-constrained environments. The sliding window implementation supports models like Mistral with local attention patterns.
