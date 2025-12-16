# Heuristic Index: unslothai_unsloth

> Tracks Heuristic pages and which pages they apply to.
> **Update IMMEDIATELY** after creating or modifying a Heuristic page.

## Pages

| Page | File | Connections | Notes |
|------|------|-------------|-------|
| unslothai_unsloth_LoRA_Rank_Selection | [→](./heuristics/unslothai_unsloth_LoRA_Rank_Selection.md) | ✅Impl:unslothai_unsloth_get_peft_model, ✅Impl:unslothai_unsloth_PatchFastRL, ✅Workflow:unslothai_unsloth_QLoRA_Finetuning, ✅Workflow:unslothai_unsloth_GRPO_Reinforcement_Learning | LoRA rank selection (8-128) based on task type and RL requirements |
| unslothai_unsloth_Quantization_Method_Selection | [→](./heuristics/unslothai_unsloth_Quantization_Method_Selection.md) | ✅Impl:unslothai_unsloth_save_pretrained_gguf, ✅Workflow:unslothai_unsloth_GGUF_Export | GGUF quantization method selection (q4_k_m, q5_k_m, etc.) |
| unslothai_unsloth_Memory_Optimization | [→](./heuristics/unslothai_unsloth_Memory_Optimization.md) | ✅Impl:unslothai_unsloth_FastLanguageModel, ✅Impl:unslothai_unsloth_save_pretrained_merged, ✅Workflow:unslothai_unsloth_QLoRA_Finetuning, ✅Workflow:unslothai_unsloth_GRPO_Reinforcement_Learning, ✅Workflow:unslothai_unsloth_GGUF_Export | 4-bit quantization, gradient checkpointing, VRAM management |
| unslothai_unsloth_RL_Hyperparameters | [→](./heuristics/unslothai_unsloth_RL_Hyperparameters.md) | ✅Impl:unslothai_unsloth_PatchFastRL, ✅Workflow:unslothai_unsloth_GRPO_Reinforcement_Learning | GRPO/DPO/PPO hyperparameters (beta, temperature, batch_size) |
| unslothai_unsloth_Mixed_Precision_Training | [→](./heuristics/unslothai_unsloth_Mixed_Precision_Training.md) | ✅Impl:unslothai_unsloth_FastLanguageModel, ✅Workflow:unslothai_unsloth_QLoRA_Finetuning | Automatic fp16/bf16 precision selection based on model dtype |
| unslothai_unsloth_Padding_Free_Training | [→](./heuristics/unslothai_unsloth_Padding_Free_Training.md) | ✅Impl:unslothai_unsloth_FastLanguageModel, ✅Workflow:unslothai_unsloth_QLoRA_Finetuning | Sample packing for 2x+ speedup on variable-length sequences |

---

**Legend:** `✅Type:Name` = page exists | `⬜Type:Name` = page needs creation
