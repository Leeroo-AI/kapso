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

**Purpose:** Implements automatic quantizer and configuration dispatching system that routes to appropriate quantization backend based on config.

**Mechanism:** Defines two main classes: `AutoQuantizationConfig` and `AutoHfQuantizer` that use registry mappings (`AUTO_QUANTIZER_MAPPING` and `AUTO_QUANTIZATION_CONFIG_MAPPING`) to dynamically instantiate the correct quantizer/config class based on the `quant_method` field (e.g., "awq", "gptq", "bitsandbytes_8bit"). Special handling exists for bitsandbytes to distinguish 4-bit vs 8-bit modes. The `merge_quantization_configs` method reconciles configs from model files with user-provided arguments. Also provides `register_quantization_config` and `register_quantizer` decorators for extensibility, plus `get_hf_quantizer` helper function that orchestrates quantizer creation and validation.

**Significance:** Core dispatch mechanism that enables a unified interface for 20+ different quantization methods. Users specify a config and the system automatically routes to the appropriate implementation (AWQ, GPTQ, BitsAndBytes, AQLM, etc.), making quantization backend-agnostic and extensible for new methods.
