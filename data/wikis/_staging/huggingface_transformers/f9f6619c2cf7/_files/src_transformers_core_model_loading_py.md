# File: `src/transformers/core_model_loading.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 1031 |
| Classes | `ConversionOps`, `Chunk`, `Concatenate`, `MergeModulelist`, `SplitModulelist`, `PermuteForRope`, `WeightTransform`, `WeightRenaming`, `WeightConverter`, `SkipParameters` |
| Functions | `build_glob_alternation`, `spawn_materialize`, `spawn_tp_materialize`, `dot_natural_key`, `log_conversion_errors`, `set_param_for_module`, `offload_and_maybe_resave_param`, `rename_source_key`, `... +2 more` |
| Imports | __future__, abc, collections, concurrent, contextlib, copy, dataclasses, integrations, os, re, ... +3 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Core infrastructure for loading model checkpoints with weight transformations, tensor parallelism, quantization, and async loading. Handles complex weight conversions during model initialization.

**Mechanism:** WeightTransform base class with WeightRenaming (simple key renaming) and WeightConverter (applies ConversionOps operations) subclasses. ConversionOps include Concatenate (merge tensors), Chunk (split tensors), MergeModulelist (stack expert weights), SplitModulelist (unstack experts), and PermuteForRope (RoPE format conversion). Uses regex patterns with named groups for flexible key matching and rename_source_key() for pattern-based renaming. Supports async loading with ThreadPoolExecutor and Future objects for materialize_tensors(). Integrates with tensor parallelism (DTensor), quantization operations, and CPU offloading. Builds reverse mappings for saving back to original format.

**Significance:** Enables advanced model loading scenarios: fused expert weights for better performance, loading safetensors/pickle/GGUF formats, tensor parallel model loading across devices, quantization during loading, and weight format migration without full model rewrites. The operation system is composable and reversible, allowing models to load from one format and save to another while maintaining compatibility.
