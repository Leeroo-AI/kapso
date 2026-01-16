# Heuristic Index: Jaymody_PicoGPT

> Tracks Heuristic pages and which pages they apply to.
> **Update IMMEDIATELY** after creating or modifying a Heuristic page.

## Pages

| Page | File | Connections | Notes |
|------|------|-------------|-------|
| Jaymody_PicoGPT_Causal_Masking_Large_Negative | [→](./heuristics/Jaymody_PicoGPT_Causal_Masking_Large_Negative.md) | ✅Impl:Jaymody_PicoGPT_Gpt2, ✅Principle:Jaymody_PicoGPT_Transformer_Architecture | Use -1e10 instead of -inf for causal mask |
| Jaymody_PicoGPT_Pre_Norm_Architecture | [→](./heuristics/Jaymody_PicoGPT_Pre_Norm_Architecture.md) | ✅Impl:Jaymody_PicoGPT_Gpt2, ✅Principle:Jaymody_PicoGPT_Transformer_Architecture | Apply LayerNorm before attention/FFN |
| Jaymody_PicoGPT_Weight_Tying_Embeddings | [→](./heuristics/Jaymody_PicoGPT_Weight_Tying_Embeddings.md) | ✅Impl:Jaymody_PicoGPT_Gpt2, ✅Principle:Jaymody_PicoGPT_Transformer_Architecture | Reuse token embeddings for output projection |
| Jaymody_PicoGPT_Stable_Softmax | [→](./heuristics/Jaymody_PicoGPT_Stable_Softmax.md) | ✅Impl:Jaymody_PicoGPT_Gpt2, ✅Principle:Jaymody_PicoGPT_Transformer_Architecture | Subtract max before exp for numerical stability |
| Jaymody_PicoGPT_Streaming_Download_Large_Files | [→](./heuristics/Jaymody_PicoGPT_Streaming_Download_Large_Files.md) | ✅Impl:Jaymody_PicoGPT_Download_Gpt2_Files, ✅Principle:Jaymody_PicoGPT_Model_Download | Stream large files with iter_content |
| Jaymody_PicoGPT_BPE_Caching_LRU | [→](./heuristics/Jaymody_PicoGPT_BPE_Caching_LRU.md) | ✅Impl:Jaymody_PicoGPT_Encoder, ✅Principle:Jaymody_PicoGPT_BPE_Tokenization | Cache BPE results with lru_cache and dict |
| Jaymody_PicoGPT_Sequence_Length_Validation | [→](./heuristics/Jaymody_PicoGPT_Sequence_Length_Validation.md) | ✅Impl:Jaymody_PicoGPT_Generate, ✅Principle:Jaymody_PicoGPT_Autoregressive_Generation | Validate input + output < n_ctx |

---

**Legend:** `✅Type:Name` = page exists | `⬜Type:Name` = page needs creation
