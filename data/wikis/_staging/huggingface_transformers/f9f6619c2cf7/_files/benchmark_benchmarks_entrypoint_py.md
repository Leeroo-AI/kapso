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

**Purpose:** Main benchmarking entrypoint for transformers CI/CD that auto-discovers and runs benchmark modules. Provides dual-mode data collection (PostgreSQL database and CSV export) with comprehensive metrics recording for model and device measurements.

**Mechanism:** Scans the `benches/` directory for Python files with `run_benchmark` functions and executes them sequentially. The `MetricsRecorder` class manages data collection, supporting both database storage (via psycopg2) and pandas-based CSV export. Dynamically imports benchmark modules and passes a global metrics recorder, supporting backward compatibility with old benchmark signatures. Exports aggregated CSV files with summary statistics (mean, max, std) for device metrics.

**Significance:** Core benchmarking infrastructure used in CI pipelines to track performance across commits. Flexible storage options ensure benchmarks can run in environments without database access (using CSV fallback). The auto-discovery pattern makes adding new benchmarks seamless, while the structured metrics recording enables historical performance analysis and regression detection.
