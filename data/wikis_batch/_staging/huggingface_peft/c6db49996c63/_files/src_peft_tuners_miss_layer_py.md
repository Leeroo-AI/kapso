# File: `src/peft/tuners/miss/layer.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 393 |
| Classes | `MissLayer`, `MissLinear` |
| Imports | __future__, math, peft, torch, typing, warnings |

## Understanding

**Status:** ✅ Documented

**Purpose:** Implements MiSS (Mixture of Sharded Squares) adapter layers using Householder reflection-based weight updates for Linear layers only.

**Mechanism:** Stores a miss_block parameter with shape (r, out_features) for balance/bat modes or (r, mini_r) for mini mode. During forward pass, reshapes input into blocks and applies transformation: result += sum(dropout(x).reshape(..., x.size(-1)//r, r), dim=-2) @ miss. The bat variant uses matrix multiplication with the identity plus miss_block, while mini/balance variants add miss_block directly to reshaped weights. Supports merge/unmerge operations with special handling for inverse computations in bat mode.

**Significance:** Implements Householder reflection adaptation, a novel approach to parameter-efficient fine-tuning. The three variants (balance, bat, mini) offer different trade-offs between parameter count, efficiency, and expressiveness. Only supports Linear layers, not convolutional operations.
