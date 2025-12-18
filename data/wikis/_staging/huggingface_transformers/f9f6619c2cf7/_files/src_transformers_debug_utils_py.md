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

**Purpose:** Provides debugging utilities to detect and diagnose numerical instability issues (NaN/Inf values) during model training, particularly helpful for identifying overflow/underflow problems in mixed-precision training.

**Mechanism:** The DebugUnderflowOverflow class registers forward hooks on all model modules to track absolute min/max values of weights, inputs, and outputs. It maintains a rolling buffer of recent frames and dumps detailed statistics when NaN or Inf is detected, showing the progression leading to the numerical instability. Supports both detection mode (automatic monitoring) and trace mode (manual batch inspection).

**Significance:** Essential debugging tool for training stability, especially critical for fp16 mixed precision training where numerical overflow is common. Helps developers quickly pinpoint which layer and operation causes training to diverge, significantly reducing debugging time for complex models. The detailed frame history shows the cascade of values leading to the problem.
