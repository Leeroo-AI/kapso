# Heuristic Index: huggingface_peft

> Tracks Heuristic pages and which pages they apply to.
> **Update IMMEDIATELY** after creating or modifying a Heuristic page.

## Pages

| Page | File | Connections | Notes |
|------|------|-------------|-------|
| huggingface_peft_Gradient_Checkpointing | [→](./heuristics/huggingface_peft_Gradient_Checkpointing.md) | ✅Impl:huggingface_peft_prepare_model_for_kbit_training, ✅Workflow:huggingface_peft_QLoRA_Training, ✅Principle:huggingface_peft_Memory_Optimization | Enable gradient checkpointing for memory savings |
| huggingface_peft_Quantized_Merge_Rounding | [→](./heuristics/huggingface_peft_Quantized_Merge_Rounding.md) | ✅Impl:huggingface_peft_merge_and_unload, ✅Workflow:huggingface_peft_Adapter_Inference, ✅Principle:huggingface_peft_Adapter_Merging | Rounding errors warning for quantized merges |
| huggingface_peft_4bit_Defensive_Clone | [→](./heuristics/huggingface_peft_4bit_Defensive_Clone.md) | ✅Impl:huggingface_peft_BitsAndBytesConfig, ✅Workflow:huggingface_peft_QLoRA_Training | Clone tensor for 4-bit backprop safety |
| huggingface_peft_DoRA_Mixed_Batch_Limitation | [→](./heuristics/huggingface_peft_DoRA_Mixed_Batch_Limitation.md) | ✅Impl:huggingface_peft_LoraConfig, ✅Workflow:huggingface_peft_Multi_Adapter_Management | DoRA incompatible with adapter_names |
| huggingface_peft_Safe_Merge_NaN_Check | [→](./heuristics/huggingface_peft_Safe_Merge_NaN_Check.md) | ✅Impl:huggingface_peft_merge_and_unload, ✅Workflow:huggingface_peft_Adapter_Inference, ✅Principle:huggingface_peft_Adapter_Merging | Use safe_merge=True to detect broken adapters |

---

## Heuristic Categories

| Category | Count | Description |
|----------|-------|-------------|
| Memory Optimization | 2 | Gradient checkpointing, tensor cloning |
| Quantization | 3 | Merge rounding, defensive clone, safe merge |
| Multi-Adapter | 1 | DoRA mixed batch limitation |
| Debugging | 2 | Safe merge NaN check, rounding warnings |

---

**Legend:** `✅Type:Name` = page exists | `⬜Type:Name` = page needs creation
