# File: `src/peft/tuners/vblora/layer.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 251 |
| Classes | `VBLoRALayer`, `Linear` |
| Imports | peft, torch, transformers, typing, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements VBLoRA adapter layers with vector bank and top-k selection mechanism.

**Mechanism:** VBLoRALayer stores trainable logits (vblora_logits_A/B) with shape (r, in_tiles, num_vectors) and (out_tiles, r, num_vectors), and shared vblora_vector_bank (num_vectors × vector_length). _get_low_rank_matrix performs top-k selection on logits, applies softmax to get weights, and computes weighted sum of selected vectors. _get_lora_matrices constructs full A (rank × in_features) and B (out_features × rank) matrices by tiling selected vectors. Forward pass: result = base + F.linear(F.linear(x, A), B).

**Significance:** Implements VBLoRA's key innovation: constructing LoRA matrices from a shared vector bank via learned top-k selection. Achieves 10-100x compression versus standard LoRA while maintaining performance. The vector bank acts as a learned dictionary, with logits determining optimal combinations per layer/rank position.
