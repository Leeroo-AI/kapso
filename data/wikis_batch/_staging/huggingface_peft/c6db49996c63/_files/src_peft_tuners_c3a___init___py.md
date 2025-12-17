# File: `src/peft/tuners/c3a/__init__.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 23 |
| Imports | config, layer, model, peft |

## Understanding

**Status:** ✅ Explored

**Purpose:** Registers C3A (Circulant Channel-Wise Convolution for Adaptation) as a PEFT method and exposes its public API.

**Mechanism:** Imports core C3A components (C3AConfig, C3ALayer, C3ALinear, C3AModel) and registers the method with PEFT's method registry using register_peft_method, making it available for model fine-tuning.

**Significance:** Entry point for the C3A tuning method, which uses block circulant matrices and Fast Fourier Transform operations to achieve parameter-efficient adaptation with fewer parameters than traditional methods.
