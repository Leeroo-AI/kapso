# File: `src/peft/tuners/fourierft/model.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 128 |
| Classes | `FourierFTModel` |
| Imports | __future__, itertools, layer, peft, re, torch, transformers, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements FourierFTModel, the main adapter model class that applies Fourier Fine-Tuning to pretrained transformers by replacing target Linear and Conv1D layers with FourierFTLinear layers.

**Mechanism:** FourierFTModel extends BaseTuner and implements _create_and_replace() to swap target layers with FourierFT-enabled versions. The method extracts layer-specific n_frequency from n_frequency_pattern using regex matching, handles both torch.nn.Linear and Conv1D layers with proper fan_in_fan_out settings, and creates FourierFTLinear modules with appropriate parameters. The _create_new_module() validates layer types and adjusts fan_in_fan_out flags based on layer type (Linear requires False, Conv1D requires True), raising errors for unsupported layer types.

**Significance:** This is the core model class for FourierFT (https://huggingface.co/papers/2405.03003), responsible for converting pretrained models into parameter-efficient Fourier-adapted versions. It integrates with PEFT's BaseTuner infrastructure to enable standard operations like saving, loading, merging, and multi-adapter support. The target module mapping uses TRANSFORMERS_MODELS_TO_FOURIERFT_TARGET_MODULES_MAPPING for architecture-specific defaults.
