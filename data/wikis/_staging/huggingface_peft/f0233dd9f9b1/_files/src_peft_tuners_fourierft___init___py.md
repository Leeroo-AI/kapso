# File: `src/peft/tuners/fourierft/__init__.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 24 |
| Imports | config, layer, model, peft |

## Understanding

**Status:** ✅ Explored

**Purpose:** Package initialization file for the FourierFT (Fourier Fine-Tuning) PEFT method that exports key components and registers FourierFT as a PEFT method in the library.

**Mechanism:** The file imports FourierFTConfig, FourierFTLayer, FourierFTLinear, and FourierFTModel from their respective modules, exposes them via __all__, and calls register_peft_method() to register FourierFT with the PEFT framework, mapping the name "fourierft" to FourierFTConfig and FourierFTModel. This registration enables users to instantiate FourierFT adapters using the standard PEFT API.

**Significance:** This is a core initialization file that makes FourierFT available as a first-class PEFT method. FourierFT is a parameter-efficient fine-tuning technique that learns weight updates in the frequency domain using Discrete Fourier Transform (DFT), requiring significantly fewer parameters than LoRA while maintaining comparable performance. The registration here integrates it into PEFT's adapter ecosystem.
