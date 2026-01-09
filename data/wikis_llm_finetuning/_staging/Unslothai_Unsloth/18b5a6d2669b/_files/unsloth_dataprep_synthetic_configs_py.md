# File: `unsloth/dataprep/synthetic_configs.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 111 |

## Understanding

**Status:** ✅ Explored

**Purpose:** Configuration template for synthetic Q&A data generation pipeline

**Mechanism:** Defines synthetic_qa_config as multi-line string containing YAML template with placeholders for paths, vLLM settings, generation parameters (temperature, top_p, chunk_size, overlap, max_tokens), cleanup thresholds, and prompts for summary generation, Q&A pair creation, and quality rating

**Significance:** Provides declarative configuration system for SyntheticDataKit that separates pipeline parameters from code, enabling users to customize data generation behavior through simple parameter substitution. The YAML structure organizes all aspects of the synthetic data workflow including input/output paths, LLM inference settings, and prompt engineering for different generation tasks
