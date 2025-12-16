# File: `unsloth/dataprep/synthetic_configs.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 111 |

## Understanding

**Status:** ✅ Explored

**Purpose:** Defines the YAML configuration template for synthetic data generation pipelines, containing paths, generation parameters, prompts, and settings for creating question-answer training data.

**Mechanism:** Contains a single multi-line string variable `synthetic_qa_config` that defines a complete YAML configuration template with placeholders for:
- Input/output paths (PDF, HTML, YouTube, DOCX, PPT, TXT inputs; parsed, generated, cleaned, final outputs)
- vLLM server configuration (API base URL, port, model name, retry settings)
- Ingest settings (default format, YouTube caption preferences)
- Generation parameters (temperature, top_p, chunk_size, overlap, max_tokens, num_pairs)
- Cleanup parameters (quality threshold, batch_size, temperature for rating)
- Format conversion settings (default format, metadata inclusion, JSON formatting)
- Prompt templates (summary generation, QA pair generation with JSON format rules, QA rating with 1-10 scale)
The template uses placeholder syntax like {model_name}, {temperature} for runtime substitution by SyntheticDataKit.prepare_qa_generation().

**Significance:** Centralizes all configuration for the synthetic data generation pipeline in a structured, human-readable format. The comprehensive prompts and parameters ensure consistent, high-quality QA pair generation across different document types and models. The template-based approach allows easy customization while maintaining sensible defaults for common use cases.
