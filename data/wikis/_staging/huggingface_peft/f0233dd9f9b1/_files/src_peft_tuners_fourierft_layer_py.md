# File: `src/peft/tuners/fourierft/layer.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 193 |
| Classes | `FourierFTLayer`, `FourierFTLinear` |
| Imports | peft, torch, transformers, typing, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements the FourierFT layer classes that learn sparse Fourier spectra and reconstruct weight deltas through inverse FFT for parameter-efficient fine-tuning.

**Mechanism:** FourierFTLayer is the base class that stores adapter parameters: fourierft_spectrum (learnable frequency coefficients), fourierft_n_frequency, fourierft_scaling, and random frequency indices. The update_layer() method randomly samples n_frequency indices from the flattened weight space, initializes spectrum parameters (zeros or random), and moves them to the correct device. The get_delta_weight() method reconstructs weight updates by: (1) creating a dense zero spectrum matrix, (2) filling sampled indices with learned values, (3) applying inverse FFT (ifft2), (4) taking the real part, and (5) scaling. FourierFTLinear extends this with forward(), merge(), and unmerge() operations. The forward pass adds F.linear(x, delta_w) to the base layer output when adapters are active.

**Significance:** These classes implement the core FourierFT algorithm (https://huggingface.co/papers/2405.03003). The key insight is that weight updates can be sparsely represented in the frequency domain - learning only n_frequency Fourier coefficients instead of all d×d weight values. This achieves extreme parameter efficiency: n_frequency=1000 uses ~16x fewer parameters than LoRA r=8. The method works because natural weight updates have sparse frequency representations, similar to how images compress well with JPEG.
