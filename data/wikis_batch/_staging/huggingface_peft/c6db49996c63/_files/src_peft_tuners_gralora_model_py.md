# File: `src/peft/tuners/gralora/model.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 142 |
| Classes | `GraloraModel` |
| Imports | __future__, layer, peft, torch, transformers, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Orchestrates GraLoRA model adaptation by replacing target linear/Conv1D layers with GraLoRA-enhanced versions.

**Mechanism:** GraloraModel extends BaseTuner, implementing _create_and_replace to substitute layers with Linear (GraLoRA's custom Linear class). Manages gralora_k, hybrid_r, and other parameters. Handles fan_in_fan_out flag for Conv1D layers with warnings. Uses TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING (shares with LoRA).

**Significance:** High-level adapter enabling block-structured low-rank fine-tuning. Transforms pretrained models into GraLoRA versions with same parameter count as LoRA (rank r) but higher expressivity due to block structure and information exchange. Supports multiple active adapters.
