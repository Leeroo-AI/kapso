# File: `method_comparison/processing.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 150 |
| Functions | `preprocess`, `load_jsons`, `load_df` |
| Imports | json, os, pandas |

## Understanding

**Status:** ✅ Explored

**Purpose:** Data processing pipeline that loads, transforms, and prepares PEFT experiment results from JSON files into structured pandas DataFrames for visualization and analysis.

**Mechanism:** The load_jsons() function reads all JSON files from a results directory. The preprocess() function extracts key metrics (memory, time, accuracy, loss, forgetting) from nested JSON structures, filters out failed experiments, and flattens the data. The load_df() function combines these steps, enforces data types, handles timestamp conversion, reorders columns for optimal viewing, and deduplicates experiments by keeping only the most recent run for each configuration.

**Significance:** Essential data transformation layer that bridges raw experiment outputs and the visualization system. By standardizing diverse metrics and handling versioning/metadata, it enables apples-to-apples comparisons across different PEFT methods, model architectures, and experimental conditions.
