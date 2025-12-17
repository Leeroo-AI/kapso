# File: `src/transformers/pipelines/base.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 1394 |
| Classes | `PipelineException`, `ArgumentHandler`, `PipelineDataFormat`, `CsvPipelineDataFormat`, `JsonPipelineDataFormat`, `PipedPipelineDataFormat`, `_ScikitCompat`, `Pipeline`, `ChunkPipeline`, `PipelineRegistry` |
| Functions | `no_collate_fn`, `pad_collate_fn`, `load_model`, `get_default_model_and_revision`, `load_assistant_model`, `build_pipeline_init_args` |
| Imports | abc, collections, contextlib, copy, csv, dynamic_module_utils, feature_extraction_utils, generation, image_processing_utils, importlib, ... +12 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Defines base classes and infrastructure for all pipeline implementations in transformers.

**Mechanism:** Pipeline abstract class provides the core three-stage pattern (preprocess/forward/postprocess) with _sanitize_parameters(), supports batching with pad_collate_fn(), device management, data format handlers (CSV/JSON/Piped), and ScikitLearn-compatible interface. ChunkPipeline extends Pipeline for processing long inputs in chunks. Includes load_model() for model instantiation, ArgumentHandler for input parsing, and PipelineRegistry for custom pipeline registration.

**Significance:** This is the foundational architecture that all 30+ task-specific pipelines inherit from, providing consistent API, batching, device placement, error handling, and extensibility while enforcing the preprocess-forward-postprocess design pattern that makes pipelines maintainable and user-friendly.
