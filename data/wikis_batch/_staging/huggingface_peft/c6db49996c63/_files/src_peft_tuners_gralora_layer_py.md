# File: `src/peft/tuners/gralora/layer.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 392 |
| Classes | `GraloraLayer`, `Linear` |
| Imports | math, peft, torch, transformers, typing, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements GraLoRA adapter layers with block-structured low-rank decomposition and optional hybrid LoRA.

**Mechanism:** GraloraLayer maintains gralora_A (N×in_features//N×rank) and gralora_B (N×rank×out_features//N) for N subblocks, plus optional gralora_A_general/gralora_B_general for hybrid mode. Forward pass scatters input across blocks, applies einsum operations with rank permutation for information exchange, then aggregates. get_delta_weight computes equivalent full weight matrix. CPU fp16/bf16 requires float32 casting.

**Significance:** Core implementation of block-wise LoRA with cross-block information flow. The scattered einsum pattern enables each subblock to access information from all other subblocks, increasing expressivity beyond standard LoRA. Hybrid mode adds unrestricted low-rank component for better adaptation flexibility.
