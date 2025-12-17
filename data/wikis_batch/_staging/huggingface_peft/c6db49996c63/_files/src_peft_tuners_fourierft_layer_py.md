# File: `src/peft/tuners/fourierft/layer.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 193 |
| Classes | `FourierFTLayer`, `FourierFTLinear` |
| Imports | peft, torch, transformers, typing, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements FourierFT adapter layers that learn sparse frequency-domain updates to linear layer weights.

**Mechanism:** FourierFTLayer stores fourierft_spectrum parameters (n_frequency trainable frequencies) and indices (random locations in frequency space). get_delta_weight reconstructs weight updates via ifft2 from sparse spectrum to spatial domain. FourierFTLinear's forward applies delta weights computed through inverse Fourier transform. Careful dtype handling (float casting) for FFT operations.

**Significance:** Core implementation of FourierFT's sparse spectral learning. Learns only n_frequency spectral coefficients instead of full weight matrices, dramatically reducing parameters. The random location seed ensures reproducible frequency selection. Supports merge/unmerge for adapter weight integration.
