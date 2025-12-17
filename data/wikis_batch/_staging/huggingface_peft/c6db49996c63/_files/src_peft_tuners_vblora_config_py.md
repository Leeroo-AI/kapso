# File: `src/peft/tuners/vblora/config.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 196 |
| Classes | `VBLoRAConfig` |
| Imports | __future__, dataclasses, peft, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Configuration dataclass for VBLoRA (Vector Bank LoRA) tuning parameters.

**Mechanism:** Defines VBLoRAConfig with key parameters: r (rank, default 4), num_vectors (256 vectors in bank), vector_length (256, must divide hidden dims), topk (2 for selection), save_only_topk_weights flag for compression, init_vector_bank_bound (0.02), and init_logits_std (0.1). Validates layers_pattern requires layers_to_transform.

**Significance:** Configures VBLoRA's unique architecture where LoRA A/B matrices are constructed by selecting top-k vectors from a shared bank using learned logits. Achieves extreme parameter efficiency - setting save_only_topk_weights=True stores only indices and weights instead of full logits, reducing storage by storing k indices per position instead of num_vectors logits. References paper https://huggingface.co/papers/2405.15179.
