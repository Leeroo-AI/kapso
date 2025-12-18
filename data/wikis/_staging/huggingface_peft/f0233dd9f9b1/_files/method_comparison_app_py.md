# File: `method_comparison/app.py`

**Category:** application

| Property | Value |
|----------|-------|
| Lines | 386 |
| Functions | `get_model_ids`, `filter_data`, `compute_pareto_frontier`, `generate_pareto_plot`, `compute_pareto_summary`, `export_csv`, `format_df`, `build_app` |
| Imports | gradio, os, plotly.express, plotly.graph_objects, processing, sanitizer, tempfile |

## Understanding

**Status:** Explored

**Purpose:** Creates and launches a Gradio web application for visualizing and comparing PEFT (Parameter-Efficient Fine-Tuning) method benchmarking results. The app displays performance metrics, generates Pareto frontier plots, and allows interactive filtering and data export.

**Mechanism:**
- Loads benchmark results from JSON files using the `processing` module
- Builds an interactive Gradio UI with dropdowns for task/model selection, data tables, and Pareto plots
- Computes Pareto frontiers to identify optimal PEFT methods across two competing metrics (e.g., memory vs accuracy)
- Uses AST-based query parsing via the `sanitizer` module to safely filter data without eval()
- Implements metric preferences dictionary that defines whether each metric should be minimized or maximized
- Creates interactive Plotly visualizations showing Pareto-dominant methods in color and dominated methods in gray
- Provides CSV export functionality for filtered results

Key functions:
- `get_model_ids()` / `filter_data()`: Filter data by task and model
- `compute_pareto_frontier()`: Identifies non-dominated solutions across two metrics
- `generate_pareto_plot()`: Creates Plotly scatter plots with Pareto frontier highlighting
- `build_app()`: Constructs the complete Gradio interface with all callbacks

**Significance:** Core component that provides the primary user interface for the PEFT benchmarking system. This is the main entry point for users to explore and compare different PEFT methods. The Pareto analysis helps researchers identify which PEFT methods offer the best trade-offs between competing objectives (e.g., accuracy vs memory usage, speed vs model size). Currently configured to display MetaMathQA benchmark results.
