# File: `src/peft/tuners/ia3/__init__.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 39 |
| Imports | config, layer, model, peft |

## Understanding

**Status:** ✅ Explored

**Purpose:** Package initialization for IA3 tuner module with dynamic quantized layer imports

**Mechanism:** Exports core IA3 components (config, model, layer classes), registers PEFT method with is_mixed_compatible=True, provides lazy loading of Linear8bitLt and Linear4bit via __getattr__ when bitsandbytes is available

**Significance:** Entry point for (IA)^3 (Infused Adapter by Inhibiting and Amplifying Inner Activations) - enables parameter-efficient fine-tuning via learned rescaling vectors on activations
