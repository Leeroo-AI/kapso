# File: `src/peft/tuners/c3a/utils.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 48 |
| Classes | `BlockCircularConvolution` |
| Functions | `get_circulant_fast` |
| Imports | torch |

## Understanding

**Status:** ✅ Explored

**Purpose:** C3A FFT-based utilities

**Mechanism:** get_circulant_fast() converts kernel to circulant matrix via FFT: creates identity, applies ifft, einsum with fft(kernel), returns fft().real.flatten(). BlockCircularConvolution (autograd Function) implements forward (einsum ifft(x) with fft(w)) and backward (compute x_grad and w_grad via FFT operations).

**Significance:** Critical FFT-based implementations for C3A. get_circulant_fast() efficiently generates circulant matrices. BlockCircularConvolution enables differentiable circular convolution with custom backward. Essential for efficient C3A operations.
