# File: `src/transformers/debug_utils.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 346 |
| Classes | `DebugUnderflowOverflow`, `DebugOption` |
| Functions | `get_abs_min_max`, `detect_overflow` |
| Imports | collections, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Debugging tool for detecting and diagnosing numerical instability (NaN/Inf) in model training, particularly useful for mixed-precision (fp16) training issues.

**Mechanism:** DebugUnderflowOverflow class registers forward hooks on all model modules to track absolute min/max values of weights, inputs, and outputs. Maintains a LIFO buffer (collections.deque) of recent frames, dumping them when NaN/Inf is detected. Supports two modes: (1) automatic detection with frame dumping on overflow, and (2) trace mode for specific batch numbers without detection. The detect_overflow function checks for NaN/Inf in tensors. Provides batch-level tracking with configurable max_frames_to_save and abort_after_batch_num for early stopping.

**Significance:** Critical debugging tool for identifying the source of training instability. When fp16 training fails with NaN losses, this tool pinpoints exactly which layer and operation produced the problematic values, showing the progression of magnitudes leading up to overflow. Particularly valuable for large models where manual inspection is impractical. The trace mode enables targeted debugging of specific problematic batches.
