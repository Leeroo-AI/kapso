# File: `gpt2_pico.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 62 |
| Functions | `gelu`, `softmax`, `layer_norm`, `linear`, `ffn`, `attention`, `mha`, `transformer_block`, `gpt2`, `generate`, `main` |
| Imports | numpy |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Minimal "pico" GPT-2 implementation.

**Mechanism:** Functionally identical to `gpt2.py` but condensed into ~40 lines of core code by removing all comments and documentation. Same architecture: GELU activation, softmax, layer normalization, linear projections, scaled dot-product attention with causal masking, multi-head attention, transformer blocks with pre-norm residuals, and autoregressive generation with greedy sampling. The code is deliberately minimized - e.g., `ffn()` is a single line, `transformer_block()` is 3 lines, `gpt2()` is 5 lines.

**Significance:** Demonstrates how compact a working GPT-2 implementation can be - the "pico" in picoGPT. Serves as an impressive demonstration that a full GPT-2 forward pass fits in ~40 lines of readable Python/NumPy. Same functionality as gpt2.py but optimized for code size rather than readability.
