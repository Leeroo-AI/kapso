# File: `src/transformers/quantizers/quantizers_utils.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 41 |
| Functions | `get_module_from_name`, `should_convert_module` |
| Imports | re, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides shared utility functions for quantizer implementations, specifically for module navigation and pattern-based module filtering.

**Mechanism:** Contains two key utility functions used across quantizers: (1) `get_module_from_name()` parses dot-separated parameter names (e.g., "model.layer.0.weight") to retrieve both the parent module and the final tensor name by splitting on the last dot and using PyTorch's `get_submodule()`, and (2) `should_convert_module()` determines whether a module should be quantized based on exclusion patterns using regex matching. The filtering logic handles three cases: exact pattern matches, pattern prefixes followed by dots, and pattern suffixes, returning False if any exclusion pattern matches (inverted logic to determine inclusion).

**Significance:** Essential shared code that prevents duplication across multiple quantizer implementations. These utilities handle common tasks needed by nearly all quantizers: navigating the model's module hierarchy to apply quantization selectively, and implementing flexible pattern-based exclusion rules for modules that should remain in full precision (like embedding layers, layer norms, or final classification heads). Centralizing this logic ensures consistent behavior across all quantization methods.
