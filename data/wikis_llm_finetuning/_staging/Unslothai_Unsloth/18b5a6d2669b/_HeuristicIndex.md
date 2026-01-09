# Heuristic Index: Unslothai_Unsloth

> Tracks Heuristic pages and which pages they apply to.
> **Update IMMEDIATELY** after creating or modifying a Heuristic page.

## Pages

| Page | File | Connections | Notes |
|------|------|-------------|-------|
| Unslothai_Unsloth_Gradient_Checkpointing_Tip | [→](./heuristics/Unslothai_Unsloth_Gradient_Checkpointing_Tip.md) | ✅Impl:Unslothai_Unsloth_FastLanguageModel_from_pretrained, ✅Impl:Unslothai_Unsloth_FastVisionModel_from_pretrained, ✅Impl:Unslothai_Unsloth_FastLanguageModel_from_pretrained_vllm | Use `use_gradient_checkpointing="unsloth"` for 50-60% VRAM reduction |
| Unslothai_Unsloth_LoRA_Rank_Selection_Tip | [→](./heuristics/Unslothai_Unsloth_LoRA_Rank_Selection_Tip.md) | ✅Impl:Unslothai_Unsloth_get_peft_model, ✅Impl:Unslothai_Unsloth_get_peft_model_rl, ✅Impl:Unslothai_Unsloth_FastLanguageModel_from_pretrained_vllm, ✅Impl:Unslothai_Unsloth_UnslothGRPOTrainer | r=16 default; r<=64 for vLLM compatibility |
| Unslothai_Unsloth_Sample_Packing_Tip | [→](./heuristics/Unslothai_Unsloth_Sample_Packing_Tip.md) | ✅Impl:Unslothai_Unsloth_SFTTrainer_train, ✅Impl:Unslothai_Unsloth_UnslothTrainingArguments | `packing=True` for >2x training speedup |
| Unslothai_Unsloth_Embedding_Learning_Rate_Tip | [→](./heuristics/Unslothai_Unsloth_Embedding_Learning_Rate_Tip.md) | ✅Impl:Unslothai_Unsloth_UnslothTrainingArguments | Use 5e-5 for embeddings vs 2e-4 for LoRA |
| Unslothai_Unsloth_GGUF_Quantization_Selection_Tip | [→](./heuristics/Unslothai_Unsloth_GGUF_Quantization_Selection_Tip.md) | ✅Impl:Unslothai_Unsloth_save_to_gguf, ✅Impl:Unslothai_Unsloth_push_to_hub_gguf, ✅Impl:Unslothai_Unsloth_ALLOWED_QUANTS | q4_k_m default; q8_0 for quality |
| Unslothai_Unsloth_BFloat16_vs_Float16_Tip | [→](./heuristics/Unslothai_Unsloth_BFloat16_vs_Float16_Tip.md) | ✅Impl:Unslothai_Unsloth_FastLanguageModel_from_pretrained, ✅Impl:Unslothai_Unsloth_FastVisionModel_from_pretrained | BFloat16 on Ampere+; Float16 on older GPUs |

---

**Legend:** `✅Type:Name` = page exists | `⬜Type:Name` = page needs creation
