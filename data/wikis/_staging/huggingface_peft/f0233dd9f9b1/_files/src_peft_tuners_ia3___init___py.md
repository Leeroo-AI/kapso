# File: `src/peft/tuners/ia3/__init__.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 39 |
| Imports | config, layer, model, peft |

## Understanding

**Status:** ✅ Explored

**Purpose:** Package initialization file for the IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations) PEFT method that exports key components, registers IA3, and provides lazy imports for quantized layer variants.

**Mechanism:** The file imports IA3Config, IA3Layer, Linear, Conv2d, Conv3d, and IA3Model from their respective modules, exposes them via __all__, and calls register_peft_method() to register IA3 with the PEFT framework (marked as is_mixed_compatible=True for mixed adapter support). It implements a __getattr__ function that lazily imports quantized variants (Linear8bitLt, Linear4bit) from the bnb module only when bitsandbytes is available, raising AttributeError for unavailable attributes. This lazy loading avoids import errors when bitsandbytes is not installed.

**Significance:** This is a core initialization file that makes IA3 available as a first-class PEFT method. IA3 (https://huggingface.co/papers/2205.05638) is an extremely parameter-efficient method that learns scalar vectors to rescale activations, requiring far fewer parameters than LoRA. The lazy import pattern for quantized layers allows IA3 to work with quantized models (8-bit, 4-bit) when bitsandbytes is available, while remaining functional without it. The is_mixed_compatible flag enables combining IA3 with other PEFT methods.
