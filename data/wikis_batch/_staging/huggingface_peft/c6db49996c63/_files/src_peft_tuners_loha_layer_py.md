# File: `src/peft/tuners/loha/layer.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 444 |
| Classes | `LoHaLayer`, `Linear`, `Conv2d`, `Conv1d`, `HadaWeight`, `HadaWeightCP` |
| Functions | `make_weight`, `make_weight_cp` |
| Imports | math, peft, torch, typing |

## Understanding

**Status:** ✅ Documented

**Purpose:** Implements LoHa (Low-Rank Hadamard Product) adapter layers for Linear, Conv1d, and Conv2d operations using efficient Hadamard product decomposition.

**Mechanism:** LoHa decomposes weight updates as a Hadamard product of two low-rank matrix products: (W1a @ W1b) * (W2a @ W2b). For convolutional layers with kernel size > 1, it can use "effective convolution" mode with additional tensor parameters (t1, t2). Custom autograd functions (HadaWeight, HadaWeightCP) compute forward passes and gradients efficiently. Supports rank dropout and module dropout during training.

**Significance:** Core implementation of the LoHa method, which offers an alternative to LoRA by using Hadamard products instead of additive low-rank updates. The efficient custom autograd functions reduce memory usage during training, and the effective_conv2d mode optimizes convolutional layer adaptation. Based on the LyCORIS implementation.
