# File: `src/peft/tuners/vblora/model.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 209 |
| Classes | `VBLoRAModel` |
| Imports | __future__, config, layer, peft, torch, transformers, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Main VBLoRA model class that adapts pretrained models using vector bank-based LoRA.

**Mechanism:** VBLoRAModel extends BaseTuner, initializes shared vblora_vector_bank (num_vectors × vector_length) with uniform distribution [-bound, +bound]. _create_and_replace wraps Linear/Conv1D layers with VBLoRA layers. get_nb_savable_parameters calculates storage: if save_only_topk_weights, counts topk_indices (using appropriate dtype: uint8/16/32/64 based on num_vectors) + topk_weights instead of full logits, achieving factor of (topk/num_vectors) compression.

**Significance:** Orchestrates VBLoRA's architecture where a single learned vector bank is shared across all adapter layers. Provides print_savable_parameters to show extreme parameter efficiency. Critical for scenarios requiring many adapters or very large models, as the vector bank amortizes cost across all layers. The save_only_topk_weights mode enables deployment at fraction of LoRA size while maintaining inference/merging capability.
