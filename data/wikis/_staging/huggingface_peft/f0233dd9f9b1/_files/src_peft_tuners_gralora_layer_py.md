# File: `src/peft/tuners/gralora/layer.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 392 |
| Classes | `GraloraLayer`, `Linear` |
| Imports | math, peft, torch, transformers, typing, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements GraLoRA layer classes that perform block-wise low-rank adaptation with information exchange between subblocks, optionally combined with vanilla LoRA for hybrid adaptation.

**Mechanism:** GraloraLayer is the base class storing adapter parameters: gralora_A/gralora_B (block-wise low-rank matrices with shape [N, features//N, rank//N]), gralora_A_general/gralora_B_general (optional vanilla LoRA components), and hyperparameters (r, alpha, gralora_k, hybrid_r, scaling). The update_layer() method validates divisibility constraints, creates N=gralora_k subblocks, initializes A matrices with Kaiming and B matrices with zeros, and stacks/transposes them into proper shapes. The Linear class implements forward() using einsum operations to compute block-wise outputs with information exchange (viewing as [N,N,subrank] and permuting), optionally adding hybrid LoRA output. The get_delta_weight() method reconstructs the full weight delta by scattering gralora_A, performing block-wise matrix multiplication, and adding the hybrid component.

**Significance:** These classes implement GraLoRA's core innovation: dividing weight matrices into N subblocks and enabling information exchange between them. Unlike standard LoRA which applies a single rank-r decomposition, GraLoRA applies N separate rank-(r/N) decompositions but reshapes and permutes to allow cross-block interactions. This multiplicatively increases expressivity by factor N while maintaining the same parameter count. The hybrid mode combines GraLoRA's structured updates with vanilla LoRA's global updates for maximum flexibility.
