# File: `src/transformers/pipelines/pt_utils.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 323 |
| Classes | `PipelineDataset`, `PipelineIterator`, `PipelineChunkIterator`, `PipelinePackIterator`, `KeyDataset`, `KeyPairDataset` |
| Imports | numpy, torch, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides PyTorch utility classes for efficient pipeline data iteration and batching. Enables pipelines to process large datasets with proper memory management and batch handling without loading everything into memory.

**Mechanism:** Implements four specialized iterator patterns: PipelineDataset wraps datasets with preprocessing, PipelineIterator handles inference with optional loader_batch_size unbatching to process batched data as individual items, PipelineChunkIterator flattens nested generators from chunked preprocessing (like mask generation), and PipelinePackIterator accumulates items until is_last flag for proper chunk boundaries. Includes smart handling of ModelOutput, tensors, tuples, and nested structures with proper dimension management for unsqueezing/expanding.

**Significance:** Critical infrastructure component enabling memory-efficient pipeline processing at scale. Essential for production deployments where datasets don't fit in memory, batch processing is required for performance, or streaming inference is needed. Underpins the ability of all transformers pipelines to handle real-world large-scale data processing scenarios efficiently.
