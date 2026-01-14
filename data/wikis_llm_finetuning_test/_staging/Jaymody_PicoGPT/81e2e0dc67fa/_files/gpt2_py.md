# File: `gpt2.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 121 |
| Functions | `gelu`, `softmax`, `layer_norm`, `linear`, `ffn`, `attention`, `mha`, `transformer_block`, `... +3 more` |
| Imports | numpy |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Full GPT-2 inference implementation in pure NumPy with detailed comments.

**Mechanism:** Implements the complete GPT-2 transformer architecture using only NumPy operations. Key functions: (1) `gelu()` - GELU activation, (2) `softmax()` - numerically stable softmax, (3) `layer_norm()` - layer normalization with learnable scale/offset, (4) `linear()` - matrix multiplication + bias, (5) `ffn()` - feed-forward network with GELU, (6) `attention()` - scaled dot-product attention with causal mask, (7) `mha()` - multi-head attention splitting Q/K/V across heads, (8) `transformer_block()` - combines attention + FFN with residual connections, (9) `gpt2()` - full forward pass with token+positional embeddings through transformer blocks, (10) `generate()` - auto-regressive token generation with greedy sampling. The `main()` function provides CLI interface via `fire`.

**Significance:** Primary entry point and educational reference - this is the well-commented, readable version of the GPT-2 implementation that serves as the main learning resource. Users run this file directly to generate text.
