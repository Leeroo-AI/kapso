# File: `benchmark/benchmarks_entrypoint.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 502 |
| Classes | `ImportModuleException`, `MetricsRecorder` |
| Functions | `parse_arguments`, `import_from_path`, `create_database_connection`, `create_global_metrics_recorder` |
| Imports | argparse, datetime, importlib, json, logging, os, pandas, sys, uuid |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Main entry point for running internal transformers benchmarks with metrics collection to PostgreSQL database and/or CSV files. This file discovers and executes benchmark modules from the benches/ directory while tracking performance metrics.

**Mechanism:** The script centers around the `MetricsRecorder` class which manages dual-mode data storage (database + CSV). It automatically discovers Python modules in the benches/ folder that implement a `run_benchmark()` function, executes them sequentially, and collects two types of metrics: device measurements (CPU/GPU utilization, memory usage) and model measurements (load times, forward pass times, generation times). The recorder uses pandas DataFrames for CSV export and psycopg2 for database operations. At completion, it generates timestamped CSV files including a comprehensive summary with aggregated statistics (mean, max, std) for device metrics. The system gracefully handles missing database connections by falling back to CSV-only mode.

**Significance:** This is the automated benchmarking infrastructure for continuous performance monitoring. It enables systematic collection of performance data across commits, providing historical tracking of model efficiency and resource usage. The dual storage approach ensures data preservation even when database connectivity fails, making it robust for CI environments and development workflows.
