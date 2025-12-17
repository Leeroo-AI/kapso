# src/transformers/model_debugging_utils.py

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides a powerful debugging context manager that traces model forward passes by capturing and serializing inputs/outputs of every module, creating hierarchical JSON traces for debugging model implementations and comparing outputs across different configurations.

**Mechanism:** The file implements a comprehensive tracing system:
- **Context manager `model_addition_debugger_context()`**: Wraps all module forward methods to record execution
- **`_attach_debugger_logic()`**: Recursively wraps forward methods of all submodules with tracing logic
- **Call tree structure**: Builds nested dictionary with module paths, inputs, outputs, and children
- **Tensor serialization (`_serialize_io()`)**: Converts tensors to JSON-serializable format with two modes:
  - `use_repr=True`: Stores tensor repr as strings (default, faster, smaller)
  - `use_repr=False`: Saves full tensors to SafeTensors files with relative paths
- **Metadata extraction**: Records tensor shape, dtype, statistics (mean, std, min, max) for float tensors
- **DTensor support**: Special handling for distributed tensors
- **Layer pruning**: `prune_intermediate_layers()` removes repetitive layer blocks for readability
- **Dual output**: Creates both FULL_TENSORS.json (with all data) and SUMMARY.json (metadata only)
- **Distributed compatibility**: `_is_rank_zero()` ensures only rank 0 writes output in multi-GPU setups

**Significance:** This debugging tool is invaluable for model developers adding new architectures to Transformers. It enables precise inspection of data flow through the model, comparison of outputs between implementations (e.g., HF vs reference), identification of numerical issues, and validation of model correctness. The structured JSON output with tensor statistics allows automated testing and regression detection, while the optional full tensor export supports detailed debugging of specific issues.
