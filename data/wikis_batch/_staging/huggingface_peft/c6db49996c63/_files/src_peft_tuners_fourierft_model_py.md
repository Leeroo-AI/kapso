# File: `src/peft/tuners/fourierft/model.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 128 |
| Classes | `FourierFTModel` |
| Imports | __future__, itertools, layer, peft, re, torch, transformers, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Orchestrates FourierFT model adaptation by replacing target layers with FourierFT-enhanced versions.

**Mechanism:** FourierFTModel extends BaseTuner, implementing _create_and_replace to substitute Linear/Conv1D layers with FourierFTLinear. Uses regex pattern matching for per-layer n_frequency customization via n_frequency_pattern. Handles fan_in_fan_out flag for Conv1D layers. Leverages TRANSFORMERS_MODELS_TO_FOURIERFT_TARGET_MODULES_MAPPING for architecture-specific defaults.

**Significance:** High-level adapter transforming pretrained models to use frequency-domain fine-tuning. Paper: https://huggingface.co/papers/2405.03003. Manages adapter lifecycle and provides warning system for layer-specific configurations (e.g., fan_in_fan_out mismatches).
