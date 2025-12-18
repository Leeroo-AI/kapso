# File: `method_comparison/processing.py`

**Category:** data processing

| Property | Value |
|----------|-------|
| Lines | 151 |
| Functions | `preprocess`, `load_jsons`, `load_df` |
| Imports | json, os, pandas |

## Understanding

**Status:** Explored

**Purpose:** Loads, preprocesses, and structures PEFT benchmarking data from JSON result files into a pandas DataFrame for analysis and visualization.

**Mechanism:**
- `load_jsons()`: Scans a directory and loads all JSON files containing experiment results
- `preprocess()`: Extracts and flattens relevant fields from nested JSON structures:
  - Filters out failed experiments (status != "success")
  - Extracts metrics: memory usage, timing, model size, accuracy, loss, trainable parameters
  - Captures metadata: versions of PEFT/transformers/datasets/torch/bitsandbytes, system info
  - Determines PEFT type from config (or "full-finetuning" if none)
  - Includes "forgetting" metric (reduction in CE loss on Wikipedia sample)
- `load_df()`: Orchestrates the loading pipeline:
  - Loads and preprocesses JSON data
  - Applies type conversions to ensure correct dtypes
  - Converts timestamps to datetime objects
  - Rounds time metrics to nearest second
  - Reorders columns to show most important metrics first
  - Deduplicates by keeping only the most recent run for each (experiment_name, model_id, peft_type, created_at) combination

**Significance:** Essential data pipeline component that transforms raw benchmark results into a structured format suitable for analysis and visualization. This module standardizes the data representation and ensures consistency across different experiment runs, making it possible to fairly compare PEFT methods.
