# Heuristic Index: huggingface_peft

> Tracks Heuristic pages and which pages they apply to.
> **Update IMMEDIATELY** after creating or modifying a Heuristic page.

## Pages

| Page | File | Connections | Notes |
|------|------|-------------|-------|
| huggingface_peft_LoRA_Rank_Selection | [→](./heuristics/huggingface_peft_LoRA_Rank_Selection.md) | ✅Impl:huggingface_peft_LoraConfig_init, ✅Workflow:huggingface_peft_LoRA_Fine_Tuning, ✅Workflow:huggingface_peft_QLoRA_Training | r=8 default; use RSLoRA at r>=32 |
| huggingface_peft_Target_Module_Selection | [→](./heuristics/huggingface_peft_Target_Module_Selection.md) | ✅Impl:huggingface_peft_LoraConfig_init, ✅Workflow:huggingface_peft_LoRA_Fine_Tuning | all-linear vs q_proj,v_proj selection |
| huggingface_peft_Quantized_Merge_Warning | [→](./heuristics/huggingface_peft_Quantized_Merge_Warning.md) | ✅Impl:huggingface_peft_merge_and_unload, ✅Impl:huggingface_peft_BitsAndBytesConfig_4bit, ✅Workflow:huggingface_peft_QLoRA_Training | 4-bit/8-bit merge rounding errors |
| huggingface_peft_Gradient_Checkpointing | [→](./heuristics/huggingface_peft_Gradient_Checkpointing.md) | ✅Impl:huggingface_peft_prepare_model_for_kbit_training, ✅Workflow:huggingface_peft_QLoRA_Training, ✅Workflow:huggingface_peft_LoRA_Fine_Tuning | Trade compute for 50-60% VRAM reduction |
| huggingface_peft_DoRA_Overhead | [→](./heuristics/huggingface_peft_DoRA_Overhead.md) | ✅Impl:huggingface_peft_LoraConfig_init, ✅Workflow:huggingface_peft_LoRA_Fine_Tuning | DoRA adds overhead; merge for inference |
| huggingface_peft_Warning_Deprecated_Bone | [→](./heuristics/huggingface_peft_Warning_Deprecated_Bone.md) | ✅Impl:huggingface_peft_BoneConfig, ✅Impl:huggingface_peft_BoneModel, ✅Impl:huggingface_peft_BoneLayer | Bone deprecated v0.19.0; migrate to MiSS |

---

## Summary

- **Total Heuristic Pages:** 6
- **Optimization Heuristics:** 3 (rank selection, gradient checkpointing, DoRA)
- **Configuration Heuristics:** 1 (target module selection)
- **Warning Heuristics:** 2 (quantized merge, Bone deprecation)

---

**Legend:** `✅Type:Name` = page exists | `⬜Type:Name` = page needs creation
