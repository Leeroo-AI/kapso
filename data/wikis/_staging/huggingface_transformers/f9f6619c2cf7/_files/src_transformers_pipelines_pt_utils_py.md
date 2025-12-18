# File: `src/transformers/pipelines/pt_utils.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 323 |
| Classes | `PipelineDataset`, `PipelineIterator`, `PipelineChunkIterator`, `PipelinePackIterator`, `KeyDataset`, `KeyPairDataset` |
| Imports | numpy, torch, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides PyTorch-specific dataset and iterator utilities for efficient batch processing, data loading, and streaming in pipeline workflows.

**Mechanism:** This module implements several specialized classes: `PipelineDataset` wraps datasets with preprocessing functions, `PipelineIterator` handles batch unrolling and applies inference functions to data loaders, `PipelineChunkIterator` manages nested iterators for chunked processing (used by ChunkPipeline), `PipelinePackIterator` accumulates chunked outputs until reaching `is_last` markers, and utility classes `KeyDataset` and `KeyPairDataset` extract specific fields from dataset items. These iterators handle complex batching scenarios including ModelOutput objects, hidden states tuples, and proper tensor dimension management when unbatching.

**Significance:** This is a critical infrastructure module that enables memory-efficient pipeline processing with batching, streaming, and chunked inference. It's essential for the pipeline framework's ability to handle large datasets, support DataLoader integration, and process data in configurable batch sizes without loading everything into memory. The chunking and packing iterators are particularly important for compute-intensive pipelines like mask generation that need to process data in smaller chunks to avoid OOM errors.
