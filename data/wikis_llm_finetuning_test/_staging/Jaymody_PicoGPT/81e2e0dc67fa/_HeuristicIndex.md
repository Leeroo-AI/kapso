# Heuristic Index: Jaymody_PicoGPT

> Tracks Heuristic pages and which pages they apply to.
> **Update IMMEDIATELY** after creating or modifying a Heuristic page.

## Pages

| Page | File | Connections | Notes |
|------|------|-------------|-------|
| Jaymody_PicoGPT_Greedy_Decoding_Tradeoffs | [→](./heuristics/Jaymody_PicoGPT_Greedy_Decoding_Tradeoffs.md) | ✅Impl:Jaymody_PicoGPT_Generate, ✅Principle:Jaymody_PicoGPT_Autoregressive_Generation | Greedy vs temperature/top-k/top-p sampling |
| Jaymody_PicoGPT_Context_Length_Limits | [→](./heuristics/Jaymody_PicoGPT_Context_Length_Limits.md) | ✅Impl:Jaymody_PicoGPT_Generate, ✅Impl:Jaymody_PicoGPT_Gpt2, ✅Principle:Jaymody_PicoGPT_Transformer_Forward_Pass | n_ctx=1024 token limit and chunking |
| Jaymody_PicoGPT_No_KV_Cache_Performance | [→](./heuristics/Jaymody_PicoGPT_No_KV_Cache_Performance.md) | ✅Impl:Jaymody_PicoGPT_Generate, ✅Impl:Jaymody_PicoGPT_Gpt2 | O(n^2) complexity without KV caching |
| Jaymody_PicoGPT_Model_Size_Memory_Requirements | [→](./heuristics/Jaymody_PicoGPT_Model_Size_Memory_Requirements.md) | ✅Impl:Jaymody_PicoGPT_Load_Encoder_Hparams_And_Params, ✅Principle:Jaymody_PicoGPT_Model_Loading | RAM/disk requirements by model size |

---

**Legend:** `✅Type:Name` = page exists | `⬜Type:Name` = page needs creation
