# File: `unsloth/dataprep/synthetic_configs.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 111 |

## Understanding

**Status:** ✅ Explored

**Purpose:** Template configuration file for synthetic data generation specifying paths, model settings, generation parameters, and LLM prompts.

**Mechanism:** Defines synthetic_qa_config string containing YAML structure with placeholders for data locations, model name, generation parameters (temperature, top_p, chunk_size, overlap, max_tokens), cleanup thresholds, and templated prompts for summary/QA generation/rating tasks. Config supports interpolation via format() method.

**Significance:** Provides flexible configuration template enabling customization of synthetic data generation behavior including model selection, text chunking strategy, and quality thresholds.
