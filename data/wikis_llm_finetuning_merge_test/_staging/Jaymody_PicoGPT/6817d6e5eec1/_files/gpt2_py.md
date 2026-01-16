# File: `gpt2.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 121 |
| Functions | `gelu`, `softmax`, `layer_norm`, `linear`, `ffn`, `attention`, `mha`, `transformer_block`, `gpt2`, `generate`, `main` |
| Imports | numpy |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Main GPT-2 implementation with full documentation.

**Mechanism:** Implements the complete GPT-2 architecture in pure NumPy (~60 lines of core code). Building blocks include: `gelu()` activation function, `softmax()` for attention weights, `layer_norm()` for normalization with gamma/beta parameters, `linear()` for matrix projections. The `attention()` function computes scaled dot-product attention with causal masking. `mha()` implements multi-head attention by splitting Q/K/V into heads, applying attention per head, then merging. `transformer_block()` combines MHA and FFN with pre-norm residual connections. The `gpt2()` function adds token+positional embeddings, runs through transformer blocks, and projects to vocabulary logits. `generate()` performs autoregressive decoding with greedy sampling. `main()` provides CLI interface via python-fire.

**Significance:** Primary entry point and reference implementation - this is the readable, well-documented version intended for educational purposes. Shows exactly how GPT-2 works with detailed comments explaining tensor shapes at each step.
