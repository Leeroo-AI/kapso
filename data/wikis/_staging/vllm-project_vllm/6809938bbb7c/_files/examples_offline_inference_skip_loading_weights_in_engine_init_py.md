# File: `examples/offline_inference/skip_loading_weights_in_engine_init.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 53 |
| Functions | `print_prompts_and_outputs`, `main` |
| Imports | vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Demonstrates deferring weight loading until after engine initialization for custom weight management patterns.

**Mechanism:** Uses skip_load_weights=True in EngineArgs to initialize model structure without loading weights. Allows manual weight loading or manipulation before first inference. Useful for custom checkpoint formats or weight preprocessing.

**Significance:** Shows advanced pattern for scenarios requiring custom weight loading logic, such as loading from non-standard formats, applying transformations, or implementing custom weight distribution strategies.
