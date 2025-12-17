# File: `src/peft/tuners/fourierft/__init__.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 24 |
| Imports | config, layer, model, peft |

## Understanding

**Status:** ✅ Explored

**Purpose:** Registers FourierFT (Fourier Fine-Tuning) as a PEFT method and exposes its public API.

**Mechanism:** Imports FourierFT components (FourierFTConfig, FourierFTLayer, FourierFTLinear, FourierFTModel) and registers the method with PEFT's method registry using register_peft_method, enabling frequency-domain fine-tuning.

**Significance:** Entry point for FourierFT, a parameter-efficient method that learns sparse updates in the frequency domain using Discrete Fourier Transform, achieving significant parameter reduction compared to LoRA while maintaining performance.
