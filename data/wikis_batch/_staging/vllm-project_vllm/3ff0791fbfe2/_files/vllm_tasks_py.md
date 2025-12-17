# File: `vllm/tasks.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 13 |
| Imports | typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Task type definitions

**Mechanism:** Defines type aliases for supported model tasks using Python's Literal types. GenerationTask includes "generate" and "transcription" for text/audio generation. PoolingTask includes "embed", "classify", "score", "token_embed", "token_classify", and "plugin" for embedding and classification tasks. SupportedTask is the union of both. Uses get_args() to create runtime-accessible task lists.

**Significance:** Central type system for model task classification. Used throughout the codebase for task-specific dispatch, validation, and configuration. Provides type safety and clear documentation of supported capabilities. Essential for routing requests to appropriate execution paths (generation vs pooling) and validating parameter compatibility.
