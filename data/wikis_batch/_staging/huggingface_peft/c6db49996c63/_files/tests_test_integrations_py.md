# File: `tests/test_integrations.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 97 |
| Classes | `MLP`, `TestInitEmptyWeights` |
| Functions | `get_mlp` |
| Imports | peft, torch |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests for integration utilities and empty weight initialization.

**Mechanism:** Tests the `init_empty_weights` context manager for creating models on meta device, and the `skip_init_on_device` decorator that prevents parameter movement to meta device. Verifies that parameters stay on the correct device (CPU vs meta) depending on context and decorators.

**Significance:** Ensures proper integration with accelerate's meta device functionality for low-memory model loading, which is critical for handling large models efficiently.
