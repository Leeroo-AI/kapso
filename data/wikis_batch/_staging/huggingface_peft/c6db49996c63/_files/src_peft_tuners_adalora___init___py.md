# File: `src/peft/tuners/adalora/__init__.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 43 |
| Imports | config, gptq, layer, model, peft |

## Understanding

**Status:** ✅ Explored

**Purpose:** Package initialization for AdaLoRA tuner module with dynamic quantized layer imports

**Mechanism:** Exports core AdaLoRA components (config, model, layers), registers the PEFT method, and provides lazy loading of quantized layer implementations (8-bit and 4-bit) via __getattr__ when bitsandbytes is available

**Significance:** Entry point for AdaLoRA (Adaptive LoRA) - enables rank-adaptive parameter-efficient fine-tuning by dynamically adjusting ranks during training based on importance scores
