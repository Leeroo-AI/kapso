# File: `src/transformers/pipelines/__init__.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 1086 |
| Functions | `get_supported_tasks`, `get_task`, `check_task`, `clean_custom_task`, `pipeline`, `pipeline`, `pipeline`, `pipeline`, `... +29 more` |
| Imports | any_to_any, audio_classification, automatic_speech_recognition, base, configuration_utils, deprecated, depth_estimation, document_question_answering, dynamic_module_utils, feature_extraction, ... +34 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Main entry point for the pipelines module that registers all available task types and provides the unified `pipeline()` factory function for creating task-specific pipelines.

**Mechanism:** Defines SUPPORTED_TASKS dictionary mapping task names to pipeline implementations and default models, TASK_ALIASES for common task name variations, imports all pipeline classes, and implements the pipeline() function that instantiates the appropriate pipeline class based on task string or auto-detects from model configuration.

**Significance:** This is the core orchestration module that makes transformers pipelines user-friendly by providing a single entry point (`pipeline()`) to access 30+ different NLP/CV/audio tasks with automatic model loading, task detection, and consistent API across all task types.
