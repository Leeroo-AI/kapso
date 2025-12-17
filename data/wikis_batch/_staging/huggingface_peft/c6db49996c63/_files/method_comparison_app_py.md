# File: `method_comparison/app.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 385 |
| Functions | `get_model_ids`, `filter_data`, `compute_pareto_frontier`, `generate_pareto_plot`, `compute_pareto_summary`, `export_csv`, `format_df`, `build_app` |
| Imports | gradio, os, plotly, processing, sanitizer, tempfile |

## Understanding

**Status:** ✅ Explored

**Purpose:** Interactive Gradio web application for visualizing and comparing PEFT method performance using Pareto frontier analysis across multiple metrics like memory usage, training time, and accuracy.

**Mechanism:** Builds a multi-page Gradio interface with dropdowns for task/model selection, filterable data tables, and interactive Plotly charts. Computes Pareto frontiers by identifying non-dominated solutions where improving one metric doesn't worsen another. Uses the sanitizer module for secure DataFrame filtering and the processing module to load experiment results. Provides CSV export functionality and hover-based detail views.

**Significance:** Core visualization tool for PEFT research that enables researchers to identify optimal method configurations by analyzing trade-offs between competing metrics. The Pareto analysis helps distinguish truly superior methods from those that only excel in one dimension while sacrificing others, making it essential for method selection and performance evaluation.
