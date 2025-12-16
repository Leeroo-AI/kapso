# Heuristic Index: unslothai_unsloth

> Tracks Heuristic pages and which pages they apply to.
> **Update IMMEDIATELY** after creating or modifying a Heuristic page.

## Pages

| Page | File | Connections | Notes |
|------|------|-------------|-------|
| unslothai_unsloth_LoRA_Rank_Selection | [→](./heuristics/unslothai_unsloth_LoRA_Rank_Selection.md) | ✅Impl:unslothai_unsloth_FastLanguageModel, ✅Impl:unslothai_unsloth_FastVisionModel, ✅Workflow:unslothai_unsloth_QLoRA_Finetuning, ✅Principle:unslothai_unsloth_Low_Rank_Adaptation | r=16 default, r=32-64 for complex tasks, lora_dropout=0 optimized |
| unslothai_unsloth_Memory_Management | [→](./heuristics/unslothai_unsloth_Memory_Management.md) | ✅Impl:unslothai_unsloth_FastLanguageModel, ✅Impl:unslothai_unsloth_UnslothTrainer, ✅Workflow:unslothai_unsloth_QLoRA_Finetuning, ✅Principle:unslothai_unsloth_Gradient_Checkpointing | maximum_memory_usage=0.9, gradient checkpointing modes, packing |
| unslothai_unsloth_GGUF_Quantization_Selection | [→](./heuristics/unslothai_unsloth_GGUF_Quantization_Selection.md) | ✅Impl:unslothai_unsloth_save_to_gguf, ✅Workflow:unslothai_unsloth_Model_Export_GGUF, ✅Principle:unslothai_unsloth_GGUF_Model_Quantization | q4_k_m default, q8_0 for quality, q2_k for size |
| unslothai_unsloth_Learning_Rate_Guidelines | [→](./heuristics/unslothai_unsloth_Learning_Rate_Guidelines.md) | ✅Impl:unslothai_unsloth_UnslothTrainer, ✅Workflow:unslothai_unsloth_QLoRA_Finetuning, ✅Principle:unslothai_unsloth_Supervised_Fine_Tuning | lr=2e-4 default, embedding_lr=5e-5 |
| unslothai_unsloth_Dtype_Selection | [→](./heuristics/unslothai_unsloth_Dtype_Selection.md) | ✅Impl:unslothai_unsloth_FastLanguageModel, ✅Impl:unslothai_unsloth_FastVisionModel, ✅Workflow:unslothai_unsloth_QLoRA_Finetuning, ✅Principle:unslothai_unsloth_QLoRA_4bit_Quantization | bfloat16 preferred on Ampere+, float32 for Gemma3/GPT-OSS |

---

**Legend:** `✅Type:Name` = page exists | `⬜Type:Name` = page needs creation
