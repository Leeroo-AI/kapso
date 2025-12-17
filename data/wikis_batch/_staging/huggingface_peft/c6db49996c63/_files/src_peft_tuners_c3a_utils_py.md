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

**Purpose:** Provides efficient FFT-based operations for circulant matrix convolution in C3A.

**Mechanism:** get_circulant_fast reconstructs full circulant matrix from compressed representation using FFT/IFFT operations. BlockCircularConvolution is a custom autograd Function implementing forward (einsum with FFT) and backward (gradient computation via FFT) passes for differentiable block circular convolution. All operations work in frequency domain for O(n log n) complexity.

**Significance:** Critical utility enabling C3A's computational efficiency. Circulant matrices can be diagonalized by Fourier transform, allowing fast matrix-vector products. The custom autograd function ensures correct gradient flow during training while maintaining performance.
