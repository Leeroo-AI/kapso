# Heuristic Index: unslothai_unsloth

> Tracks Heuristic pages and which pages they apply to.
> **Update IMMEDIATELY** after creating or modifying a Heuristic page.

## Pages

| Page | File | Connections | Notes |
|------|------|-------------|-------|
| unslothai_unsloth_Import_Order | [→](./heuristics/unslothai_unsloth_Import_Order.md) | ✅Impl:unslothai_unsloth_import_unsloth, ✅Workflow:unslothai_unsloth_QLoRA_Finetuning, ✅Workflow:unslothai_unsloth_GRPO_Reinforcement_Learning | Import unsloth BEFORE transformers/trl/peft |
| unslothai_unsloth_Gradient_Checkpointing | [→](./heuristics/unslothai_unsloth_Gradient_Checkpointing.md) | ✅Impl:unslothai_unsloth_get_peft_model, ✅Impl:unslothai_unsloth_FastLanguageModel_from_pretrained, ✅Workflow:unslothai_unsloth_QLoRA_Finetuning, ✅Workflow:unslothai_unsloth_GRPO_Reinforcement_Learning, ✅Principle:unslothai_unsloth_LoRA_Configuration | use_gradient_checkpointing="unsloth" for 30% VRAM reduction |
| unslothai_unsloth_Sample_Packing | [→](./heuristics/unslothai_unsloth_Sample_Packing.md) | ✅Impl:unslothai_unsloth_SFTTrainer_usage, ✅Impl:unslothai_unsloth_trainer_train, ✅Workflow:unslothai_unsloth_QLoRA_Finetuning, ✅Principle:unslothai_unsloth_Training_Configuration | packing=True for >2x faster training |
| unslothai_unsloth_AMD_GPU_Limitations | [→](./heuristics/unslothai_unsloth_AMD_GPU_Limitations.md) | ✅Impl:unslothai_unsloth_FastLanguageModel_from_pretrained, ✅Principle:unslothai_unsloth_Model_Loading, ✅Env:unslothai_unsloth_CUDA | AMD ROCm blocksize differences and bitsandbytes compatibility |
| unslothai_unsloth_Flash_Attention_Gemma2 | [→](./heuristics/unslothai_unsloth_Flash_Attention_Gemma2.md) | ✅Impl:unslothai_unsloth_FastLanguageModel_from_pretrained, ✅Principle:unslothai_unsloth_Model_Loading, ✅Env:unslothai_unsloth_CUDA | flash-attn>=2.6.3 for Gemma 2 softcapping |

---

**Legend:** `✅Type:Name` = page exists | `⬜Type:Name` = page needs creation
