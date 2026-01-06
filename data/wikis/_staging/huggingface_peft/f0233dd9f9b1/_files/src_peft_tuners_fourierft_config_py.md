# File: `src/peft/tuners/fourierft/config.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 206 |
| Classes | `FourierFTConfig` |
| Imports | __future__, dataclasses, peft, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Defines FourierFTConfig, the configuration dataclass for Fourier Fine-Tuning (FourierFT), which specifies hyperparameters for learning weight updates in the frequency domain via sparse Fourier spectra.

**Mechanism:** FourierFTConfig stores key parameters: (1) n_frequency - the number of learnable frequency components (typically 1000-3000), which determines parameter count and model expressivity; (2) scaling - a scaling factor similar to LoRA's alpha (typically 100-300); (3) random_loc_seed - seed for random frequency location selection; (4) Standard PEFT parameters like target_modules, layers_to_transform, and bias settings; (5) init_weights - whether to initialize spectrum to zeros (True) or standard normal (False); (6) n_frequency_pattern - allows per-layer frequency configuration. The __post_init__ validates that layers_to_transform and layers_pattern are only used with list-based target_modules, not regex patterns.

**Significance:** This configuration is essential for FourierFT implementation (https://huggingface.co/papers/2405.03003). FourierFT achieves parameter efficiency by learning a sparse Fourier spectrum and reconstructing weight deltas via inverse FFT. With n_frequency=1000, it uses ~16x fewer parameters than LoRA r=8 while achieving similar accuracy. The method is particularly effective for NLU tasks (RoBERTa), vision tasks (ViT), and instruction tuning (LLaMA).
