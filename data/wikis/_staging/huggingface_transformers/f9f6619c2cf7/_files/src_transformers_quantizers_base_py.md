# File: `src/transformers/quantizers/base.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 354 |
| Classes | `HfQuantizer`, `SequentialLlama4TextExperts` |
| Functions | `get_keys_to_not_convert` |
| Imports | abc, quantizers_utils, typing, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Defines the abstract base class and common infrastructure for all quantization methods in transformers.

**Mechanism:** The `HfQuantizer` abstract class provides a complete quantization lifecycle framework with hooks for: environment validation, device map updates, dtype adjustments, model preprocessing before/after weight loading, parameter quantization checks, dequantization, and serialization. It includes utility methods like `get_modules_to_not_convert` that identifies layers to skip (tied weights, output embeddings, last module) and `param_element_size` for memory calculation. The `_convert_model_for_quantization` method patches special modules (e.g., Llama4TextExperts) for certain quantization methods. Properties like `is_trainable`, `is_serializable`, and `is_qat_trainable` define backend capabilities.

**Significance:** Foundational abstract base class that standardizes how all 20+ quantization backends integrate with transformers. Provides the contract and shared utilities that ensure consistent behavior across different quantization methods (GPTQ, AWQ, BitsAndBytes, etc.) during model loading, inference, and serialization.
