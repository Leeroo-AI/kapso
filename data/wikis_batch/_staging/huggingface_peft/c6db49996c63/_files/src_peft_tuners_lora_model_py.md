# File: `src/peft/tuners/lora/model.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 872 |
| Classes | `LoraModel` |
| Imports | __future__, aqlm, awq, config, contextlib, dataclasses, eetq, functools, gptq, hqq, ... +11 more |

## Understanding

**Status:** ✅ Documented

**Purpose:** LoraModel class - orchestrates LoRA adapter injection and management at model level

**Mechanism:** Extends BaseTuner to inject LoRA layers into target modules of a base model. Handles adapter creation via _create_new_module (dispatches to GPTQ/AWQ/AQLM/BNB/EETQ/HQQ/INC/Torchao/Megatron/default backends), manages multiple adapters simultaneously, implements add_weighted_adapter for merging adapters with various combination strategies (linear, SVD, ties, dare_ties, magnitude_prune), and provides forward hooks for mixed-batch inference and aLoRA token routing.

**Significance:** The orchestration layer that transforms any HuggingFace model into a LoRA-enabled model. Handles complex logic for multi-adapter training/inference, adapter arithmetic, quantization backend selection, gradient checkpointing compatibility, and ensures correct device placement. This is the class users interact with when calling get_peft_model().
