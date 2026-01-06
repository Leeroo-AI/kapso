# File: `src/transformers/modelcard.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 767 |
| Classes | `ModelCard`, `TrainingSummary` |
| Functions | `infer_metric_tags_from_eval_results`, `is_hf_dataset`, `parse_log_history`, `make_markdown_table`, `extract_hyperparameters_from_trainer` |
| Imports | copy, dataclasses, httpx, huggingface_hub, json, models, os, pathlib, training_args, typing, ... +3 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Automated model card generation system that creates standardized documentation following the Model Cards paper, integrating training metadata and Hub requirements.

**Mechanism:** Two main classes: (1) ModelCard (deprecated) provides structured storage for model documentation sections (model_details, intended_use, factors, metrics, etc.) with save/load capabilities, and (2) TrainingSummary generates markdown model cards automatically from Trainer state. TrainingSummary.from_trainer() extracts hyperparameters, dataset info, evaluation metrics, and training logs, then to_model_card() renders these into a formatted markdown file with YAML frontmatter containing Hub metadata. Includes utilities for parsing trainer logs (parse_log_history), creating markdown tables (make_markdown_table), inferring task types from model classes, and extracting hyperparameters. Automatically infers license from base model and creates model-index entries for Hub leaderboards.

**Significance:** Ensures consistent, high-quality documentation across models on the Hub by automating model card creation during training. Implements the Model Cards for Model Reporting framework (Mitchell et al. 2019), promoting transparency and responsible AI practices. Critical for Hub integration and model discoverability through proper metadata tagging.
