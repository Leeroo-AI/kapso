# File: `src/transformers/conversion_mapping.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 274 |
| Functions | `get_checkpoint_conversion_mapping`, `register_checkpoint_conversion_mapping`, `get_model_conversion_mapping` |
| Imports | __future__, copy, core_model_loading, typing, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Central registry for checkpoint weight conversion mappings. Defines how to transform weights between different checkpoint formats, particularly for models using fused experts or different tensor layouts.

**Mechanism:** _build_checkpoint_conversion_mapping() creates a dictionary mapping model types to lists of WeightConverter and WeightRenaming objects. Common pattern: MoE models (Mixtral, Qwen2-MoE, PhiMoE) merge expert weights using MergeModulelist then concatenate gate/up projections. Legacy mappings handle old LayerNorm naming (gamma/beta → weight/bias) and weight_norm parametrizations. VLMS list specifies vision-language models that use _checkpoint_conversion_mapping attribute. get_model_conversion_mapping() collects all applicable conversions for a given model, including quantizer-specific conversions.

**Significance:** Enables loading checkpoints with different weight organizations without code duplication. Critical for: fused expert implementations that improve performance by batching operations, maintaining compatibility with legacy checkpoints, supporting different quantization formats, and allowing models to migrate between weight layouts. The mapping system is extensible via register_checkpoint_conversion_mapping() for custom models.
