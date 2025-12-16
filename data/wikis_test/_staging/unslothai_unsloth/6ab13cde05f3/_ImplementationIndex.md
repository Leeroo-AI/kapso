# Implementation Index: unslothai_unsloth

> Tracks Implementation pages and their connections to Principles, Environments, etc.
> **Update IMMEDIATELY** after creating or modifying a Implementation page.

## Pages

| Page | File | Connections | Source | Notes |
|------|------|-------------|--------|-------|
| unslothai_unsloth_FastLanguageModel | [→](./implementations/unslothai_unsloth_FastLanguageModel.md) | ✅Principle:unslothai_unsloth_Model_Loading, ✅Env:unslothai_unsloth_CUDA, ✅Heuristic:unslothai_unsloth_Memory_Optimization, ✅Heuristic:unslothai_unsloth_Mixed_Precision_Training | loader.py:L120-620 | Unified model loading with 4-bit quantization |
| unslothai_unsloth_get_peft_model | [→](./implementations/unslothai_unsloth_get_peft_model.md) | ✅Principle:unslothai_unsloth_LoRA_Injection, ✅Env:unslothai_unsloth_CUDA, ✅Heuristic:unslothai_unsloth_LoRA_Rank_Selection | llama.py:L2578-2800 | LoRA adapter injection with fused kernels |
| unslothai_unsloth_save_pretrained_merged | [→](./implementations/unslothai_unsloth_save_pretrained_merged.md) | ✅Principle:unslothai_unsloth_Weight_Merging, ✅Env:unslothai_unsloth_CUDA, ✅Heuristic:unslothai_unsloth_Memory_Optimization | save.py:L228-506 | Merge LoRA weights and save to HuggingFace format |
| unslothai_unsloth_save_pretrained_gguf | [→](./implementations/unslothai_unsloth_save_pretrained_gguf.md) | ✅Principle:unslothai_unsloth_GGUF_Conversion, ✅Env:unslothai_unsloth_llama_cpp, ✅Env:unslothai_unsloth_CUDA, ✅Heuristic:unslothai_unsloth_Quantization_Method_Selection | save.py:L1776-2000 | Export to GGUF format with quantization |
| unslothai_unsloth_FastVisionModel | [→](./implementations/unslothai_unsloth_FastVisionModel.md) | ✅Principle:unslothai_unsloth_Vision_Model_Loading, ✅Env:unslothai_unsloth_CUDA | loader.py:L1257, vision.py | Vision-language model loading |
| unslothai_unsloth_PatchFastRL | [→](./implementations/unslothai_unsloth_PatchFastRL.md) | ✅Principle:unslothai_unsloth_RL_Setup, ✅Env:unslothai_unsloth_vLLM, ✅Env:unslothai_unsloth_CUDA, ✅Heuristic:unslothai_unsloth_RL_Hyperparameters, ✅Heuristic:unslothai_unsloth_LoRA_Rank_Selection | rl.py:L1343-1349 | TRL trainer patching for RL optimization |

---

**Legend:** `✅Type:Name` = page exists | `⬜Type:Name` = page needs creation
