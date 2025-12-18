# File: `src/transformers/cache_utils.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 1402 |
| Classes | `CacheLayerMixin`, `DynamicLayer`, `DynamicSlidingWindowLayer`, `StaticLayer`, `StaticSlidingWindowLayer`, `QuantizedLayer`, `QuantoQuantizedLayer`, `HQQQuantizedLayer`, `Cache`, `DynamicCache`, `StaticCache`, `QuantizedCache`, `EncoderDecoderCache`, `SlidingWindowLayer`, `ChunkedSlidingLayer`, `OffloadedCache`, `OffloadedStaticCache`, `SlidingWindowCache`, `HybridCache`, `HybridChunkedCache`, `OffloadedHybridCache`, `QuantoQuantizedCache`, `HQQQuantizedCache` |
| Imports | abc, collections, configuration_utils, torch, typing, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements key-value cache management for transformer models during generation. Provides multiple cache strategies (dynamic, static, sliding window, quantized) to optimize memory usage and enable efficient autoregressive generation.

**Mechanism:** Defines abstract CacheLayerMixin base with concrete implementations: DynamicLayer grows as tokens generate, StaticLayer pre-allocates fixed size for torch.compile, DynamicSlidingWindowLayer maintains only recent tokens, and QuantizedLayer compresses cached states using quantization (Quanto/HQQ backends). The Cache container manages per-layer caches with optional offloading to CPU. Supports both standard decoder-only and encoder-decoder architectures via EncoderDecoderCache. Uses lazy initialization to infer shapes at runtime and mark_static_address for compile compatibility.

**Significance:** Critical for transformer generation performance and memory efficiency. Without caching, each token generation would recompute all previous tokens' key-values. Different cache types enable different use cases: DynamicCache for flexible generation, StaticCache for compiled models, sliding windows for long sequences, and quantization for memory-constrained devices. The architecture supports hybrid models mixing different cache strategies per layer.
