# File: `benchmark/optimum_benchmark_wrapper.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 20 |
| Functions | `main` |
| Imports | argparse, subprocess |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Thin wrapper script that invokes the external optimum-benchmark CLI tool with standardized logging configuration. This provides a consistent interface for running optimum-benchmark from within transformers.

**Mechanism:** The `main()` function constructs and executes a subprocess call to the optimum-benchmark command-line tool, passing through config directory, config name, and additional arguments. It automatically injects Hydra logging flags (`hydra/job_logging=disabled`, `hydra/hydra_logging=disabled`) to suppress verbose Hydra framework output. The script uses `argparse` to parse known arguments (config-dir, config-name) and forwards all remaining unknown arguments directly to optimum-benchmark via subprocess.run().

**Significance:** This wrapper exists to standardize how transformers invokes optimum-benchmark, ensuring consistent logging behavior across all benchmark executions. It decouples the transformers codebase from direct optimum-benchmark API dependencies, making it easier to update optimum-benchmark versions and maintain a stable benchmark interface. The logging suppression is important for clean benchmark output when running in CI or batch scenarios.
