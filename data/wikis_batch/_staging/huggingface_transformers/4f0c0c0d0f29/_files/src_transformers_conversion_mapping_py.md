# File: `src/transformers/conversion_mapping.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 274 |
| Functions | `get_checkpoint_conversion_mapping`, `register_checkpoint_conversion_mapping`, `get_model_conversion_mapping` |
| Imports | __future__, copy, core_model_loading, typing, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Defines checkpoint weight conversion mappings for transforming model weights between different checkpoint formats and merged weight schemes (especially for MoE models).

**Mechanism:** Uses _build_checkpoint_conversion_mapping to create a registry mapping model types to WeightConverter/WeightRenaming operations. Primary use case is merging expert weights in Mixture-of-Experts models (Mixtral, Qwen2-MoE, etc.) where separate expert weights (w1, w2, w3) are concatenated into unified tensors (gate_up_proj, down_proj) using MergeModulelist and Concatenate operations. Supports legacy weight name conversions (LayerNorm.gamma → LayerNorm.weight) and parametrization renaming. Includes VLMS list for vision-language models with custom checkpoint conversion mappings.

**Significance:** Enables efficient model loading by transforming checkpoint formats at load time rather than storing redundant weight copies. Critical for MoE models where merged expert weights improve memory layout and inference speed. The conversion system allows models to evolve their weight layout while maintaining backward compatibility with existing checkpoints.
