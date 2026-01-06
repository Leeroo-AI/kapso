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

**Purpose:** Minimal wrapper that invokes the optimum-benchmark CLI tool with hydra logging disabled. Provides a Python function interface to the external optimum-benchmark command-line tool.

**Mechanism:** Uses subprocess.run to execute the `optimum-benchmark` command with specified config directory and name, automatically appending hydra logging disablement flags (`hydra/job_logging=disabled`, `hydra/hydra_logging=disabled`) to reduce verbose output. Forwards all additional arguments unchanged to the underlying CLI.

**Significance:** Thin abstraction layer that allows benchmark.py to programmatically control optimum-benchmark runs while suppressing unnecessary hydra logging output. The wrapper pattern enables integration with transformers' benchmark orchestration system while maintaining clean separation from the external optimum-benchmark library.
