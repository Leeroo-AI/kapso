# File: `src/peft/tuners/gralora/model.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 142 |
| Classes | `GraloraModel` |
| Imports | __future__, layer, peft, torch, transformers, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements GraloraModel, the main adapter model class that applies GraLoRA (Granular Low-Rank Adaptation) to pretrained transformers by replacing target Linear and Conv1D layers with GraLoRA-enabled Linear layers.

**Mechanism:** GraloraModel extends BaseTuner and implements _create_and_replace() to swap target layers with GraLoRA versions. The method extracts adapter parameters (r, alpha, gralora_dropout, gralora_k, hybrid_r, init_weights) from the config and passes them to either update_layer() for existing Linear modules or _create_new_module() for new replacements. The _create_new_module() validates layer types (torch.nn.Linear or Conv1D) and adjusts fan_in_fan_out flags appropriately (False for Linear, True for Conv1D), raising errors for unsupported layer types. It creates a new Linear layer with all GraLoRA parameters including the module name for tracking.

**Significance:** This is the core model class for GraLoRA (Vector-based Random Matrix Adaptation), responsible for converting pretrained models into parameter-efficient GraLoRA-adapted versions. It integrates with PEFT's BaseTuner infrastructure for standard operations. The target module mapping uses TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING since GraLoRA targets the same modules as standard LoRA but applies block-wise decomposition with information exchange for higher expressivity.
