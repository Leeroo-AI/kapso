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

**Purpose:** Defines the abstract base class `HfQuantizer` that establishes the contract and common functionality for all quantization implementations. This file provides the foundation for the quantization system's plugin architecture.

**Mechanism:** The `HfQuantizer` abstract class defines lifecycle hooks (`preprocess_model()`, `postprocess_model()`, `_process_model_before_weight_loading()`, `_process_model_after_weight_loading()`), environment validation methods, device map management, dtype adjustment, parameter checking, and serialization/training capability flags. Includes helper utilities like `get_keys_to_not_convert()` which identifies model components that should remain unquantized (tied weights, output embeddings, last module). The `SequentialLlama4TextExperts` class provides specialized handling for MoE (Mixture of Experts) quantization in Llama4 models.

**Significance:** Core architectural component that enables the extensible quantization system. All 20+ quantization methods inherit from this base class, ensuring consistent behavior and interfaces. The abstract methods enforce implementation requirements while providing sensible defaults, making it straightforward to add new quantization methods.
