# File: `unsloth/dataprep/synthetic_configs.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 111 |

## Understanding

**Status:** ✅ Explored

**Purpose:** Defines YAML configuration template for synthetic data generation pipeline with placeholders for runtime parameters.

**Mechanism:** Single module-level string `synthetic_qa_config` containing YAML template with placeholder variables (e.g., `{data_output_location}`, `{model_name}`, `{temperature}`, etc.) that are replaced via `.replace()` calls in `SyntheticDataKit.prepare_qa_generation()`. Template includes: (1) paths configuration for input formats (pdf, html, youtube, docx, ppt, txt) and output stages (parsed, generated, cleaned, final), (2) vLLM server settings (API base URL, port, model, retries), (3) ingest configuration (default format, YouTube caption preference), (4) LLM generation parameters (temperature, top_p, chunk_size, overlap, max_tokens, num_pairs), (5) content cleanup parameters (quality threshold, batch_size, temperature), (6) format conversion settings (default to jsonl with metadata), (7) prompt templates for summary generation, QA pair generation (with JSON format specification and rules), and QA pair rating (1-10 scale with JSON output format).

**Significance:** Provides declarative configuration system for multi-stage synthetic data pipeline, separating concerns between data preparation logic and generation prompts. The prompt templates encode best practices for QA generation (questions about important facts, answers supported by text, strict JSON formatting) and quality assessment (rating scale). The chunking and overlap parameters ensure efficient processing of long documents while maintaining context. The multi-format input support (PDF, HTML, YouTube, Office formats) enables diverse data sources. The staged output paths (parsed -> generated -> cleaned -> final) reflect a quality control workflow. The temperature differentiation (0.7 for generation, 0.3 for rating) balances creativity with consistency.
