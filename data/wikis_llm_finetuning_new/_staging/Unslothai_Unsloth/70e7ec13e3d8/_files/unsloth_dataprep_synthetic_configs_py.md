# File: `unsloth/dataprep/synthetic_configs.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 111 |

## Understanding

**Status:** ✅ Explored

**Purpose:** Defines YAML configuration templates for synthetic data generation, specifically for QA pair generation and cleanup.

**Mechanism:** Contains the `synthetic_qa_config` string constant, a YAML template with placeholder variables (e.g., `{data_output_location}`, `{model_name}`, `{temperature}`, `{max_tokens}`) that define the configuration structure. The template specifies: (1) input/output paths for various document formats (PDF, HTML, YouTube, DOCX, PPT, TXT), (2) vLLM server settings (API base URL, port, retry configuration), (3) generation parameters (temperature, top_p, chunk_size, overlap, max_tokens, num_pairs), (4) cleanup parameters for quality filtering (threshold, batch_size, temperature), (5) output format settings (JSONL with metadata), and (6) prompt templates for summary generation, QA pair generation with JSON output format, and QA pair quality rating.

**Significance:** Centralizes configuration for the synthetic data generation pipeline. The template-based approach allows `SyntheticDataKit.prepare_qa_generation()` to generate customized configuration files by replacing placeholders with runtime parameters, ensuring consistent and reproducible synthetic data generation workflows.
