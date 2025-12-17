# File: `src/transformers/quantizers/auto.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 338 |
| Classes | `AutoQuantizationConfig`, `AutoHfQuantizer` |
| Functions | `register_quantization_config`, `register_quantizer`, `get_hf_quantizer` |
| Imports | base, models, quantizer_aqlm, quantizer_auto_round, quantizer_awq, quantizer_bitnet, quantizer_bnb_4bit, quantizer_bnb_8bit, quantizer_compressed_tensors, quantizer_eetq, ... +14 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements the auto-dispatch system for quantization, automatically selecting and instantiating the appropriate quantizer and configuration based on the quantization method specified. This file serves as the central registry and factory for all quantization methods.

**Mechanism:** Maintains two dictionaries (`AUTO_QUANTIZER_MAPPING` and `AUTO_QUANTIZATION_CONFIG_MAPPING`) that map quantization method names (like "awq", "gptq", "bitsandbytes") to their respective quantizer classes and configuration classes. The `AutoQuantizationConfig` class dispatches configuration loading, while `AutoHfQuantizer` instantiates the correct quantizer. Provides decorator functions `register_quantization_config()` and `register_quantizer()` for extensibility. The `get_hf_quantizer()` function orchestrates the entire quantization setup including config merging, environment validation, and device map updates.

**Significance:** Essential infrastructure that enables the unified quantization API across 20+ different quantization methods (AWQ, GPTQ, BitsAndBytes, AQLM, etc.). This abstraction allows users to load quantized models without knowing implementation details, making quantization accessible and consistent across the Transformers ecosystem.
