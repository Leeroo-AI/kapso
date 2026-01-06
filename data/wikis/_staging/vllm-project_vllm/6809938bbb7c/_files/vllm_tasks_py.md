# File: `vllm/tasks.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 13 |
| Imports | typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Type definitions for supported model task types.

**Mechanism:** Defines two main task categories using Python's `Literal` type hints: (1) `GenerationTask` includes "generate" and "transcription" for text/audio generation tasks, (2) `PoolingTask` includes "embed", "classify", "score", "token_embed", "token_classify", and "plugin" for embedding and classification tasks. `SupportedTask` is a union of both categories. Uses `get_args()` to extract task lists as tuples for runtime validation.

**Significance:** Provides type safety and clear task taxonomy for vLLM. These types are used throughout the codebase for task-specific logic branching and validation. Having centralized task definitions prevents typos and ensures consistency. The distinction between generation and pooling tasks reflects vLLM's support for different model types beyond just language generation. This simple file has wide-reaching impact on code organization and type checking.
