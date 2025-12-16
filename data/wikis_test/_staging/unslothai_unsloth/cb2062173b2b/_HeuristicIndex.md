# Heuristic Index: unslothai_unsloth

> Tracks Heuristic pages and which pages they apply to.
> **Update IMMEDIATELY** after creating or modifying a Heuristic page.

## Pages

| Page | File | Connections | Notes |
|------|------|-------------|-------|
| unslothai_unsloth_LoRA_Rank_Selection | [→](./heuristics/unslothai_unsloth_LoRA_Rank_Selection.md) | ✅Impl:unslothai_unsloth_get_peft_model, ✅Workflow:unslothai_unsloth_QLoRA_Finetuning, ✅Principle:unslothai_unsloth_LoRA_Configuration | r=8-16 simple, r=16-32 standard, r=32-64 complex |
| unslothai_unsloth_Quantization_Method_Selection | [→](./heuristics/unslothai_unsloth_Quantization_Method_Selection.md) | ✅Impl:unslothai_unsloth_save_pretrained_gguf, ✅Workflow:unslothai_unsloth_GGUF_Export, ✅Principle:unslothai_unsloth_GGUF_Conversion | q4_k_m default, q8_0 fast, q2_k smallest |
| unslothai_unsloth_Gradient_Checkpointing | [→](./heuristics/unslothai_unsloth_Gradient_Checkpointing.md) | ✅Impl:unslothai_unsloth_FastLanguageModel, ✅Workflow:unslothai_unsloth_QLoRA_Finetuning, ✅Principle:unslothai_unsloth_Environment_Setup | use_gradient_checkpointing="unsloth" for 30% VRAM savings |
| unslothai_unsloth_Memory_Management | [→](./heuristics/unslothai_unsloth_Memory_Management.md) | ✅Impl:unslothai_unsloth_save_pretrained_merged, ✅Impl:unslothai_unsloth_FastLanguageModel, ✅Workflow:unslothai_unsloth_QLoRA_Finetuning | maximum_memory_usage=0.9, layer-by-layer processing |
| unslothai_unsloth_LoRA_Dropout_Bias | [→](./heuristics/unslothai_unsloth_LoRA_Dropout_Bias.md) | ✅Impl:unslothai_unsloth_get_peft_model, ✅Principle:unslothai_unsloth_LoRA_Configuration | lora_dropout=0, bias="none" for fast patching |
| unslothai_unsloth_Sample_Packing | [→](./heuristics/unslothai_unsloth_Sample_Packing.md) | ✅Workflow:unslothai_unsloth_QLoRA_Finetuning, ✅Principle:unslothai_unsloth_SFT_Training | packing=True for >2x faster training |
| unslothai_unsloth_RL_Learning_Rate | [→](./heuristics/unslothai_unsloth_RL_Learning_Rate.md) | ✅Workflow:unslothai_unsloth_GRPO_Training, ✅Principle:unslothai_unsloth_GRPO_Training | Use 5e-6 for RL vs 2e-4 for SFT |

---

**Legend:** `✅Type:Name` = page exists | `⬜Type:Name` = page needs creation
