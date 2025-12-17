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

**Purpose:** Demonstrates deferred weight loading

**Mechanism:** Initializes LLM with load_format='dummy' for fast startup without real weights (generates dummy outputs), then calls collective_rpc to update_config with load_format='auto' and reload_weights to load actual model weights. Verifies outputs improve after loading real weights.

**Significance:** Example demonstrating lazy weight loading pattern for faster initialization when weights aren't immediately needed.
