# File: `src/peft/tuners/fourierft/config.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 206 |
| Classes | `FourierFTConfig` |
| Imports | __future__, dataclasses, peft, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Defines configuration parameters for FourierFT (Fourier Fine-Tuning) method operating in frequency domain.

**Mechanism:** FourierFTConfig extends PeftConfig with key parameters: n_frequency (number of learnable frequencies, typically 1000-3000), scaling (like lora_alpha, typically 100-300), random_loc_seed (for spectral entry locations), target_modules, bias settings, and per-layer frequency customization via n_frequency_pattern. Validates constraints in __post_init__.

**Significance:** Core configuration controlling FourierFT's frequency-domain adaptation. The n_frequency parameter determines parameter count - with same target_modules, LoRA has (2*d*r/n_frequency) times more parameters. Paper reference: https://huggingface.co/papers/2405.03003. Supports both NLU (RoBERTa) and vision (ViT) tasks with task-specific hyperparameter guidance.
