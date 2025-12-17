# File: `src/peft/peft_model.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 3387 |
| Classes | `PeftModel`, `PeftModelForSequenceClassification`, `PeftModelForCausalLM`, `PeftModelForSeq2SeqLM`, `PeftModelForTokenClassification`, `PeftModelForQuestionAnswering`, `PeftModelForFeatureExtraction`, `TunerLayerStatus`, `TunerModelStatus` |
| Functions | `get_layer_status`, `get_model_status` |
| Imports | __future__, accelerate, collections, config, contextlib, copy, dataclasses, huggingface_hub, inspect, mapping, ... +9 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Core model wrapper classes that add PEFT adapter functionality to Transformers models.

**Mechanism:** PeftModel wraps a base model and manages adapters (add, load, save, delete, activate). Handles both tuner-based methods (LoRA, etc.) and prompt learning (P-Tuning, Prefix Tuning). Task-specific subclasses override forward() to inject virtual tokens for prompt learning or pass through to base model for tuners. Includes comprehensive save/load, device management, gradient checkpointing support, and status inspection utilities.

**Significance:** The heart of the PEFT library. This is the primary interface users interact with. Handles the complex orchestration of adapters with base models, manages device placement for distributed training, integrates with Hugging Face Hub, and ensures compatibility across different model architectures and tasks. One of the largest and most critical files in the codebase.
