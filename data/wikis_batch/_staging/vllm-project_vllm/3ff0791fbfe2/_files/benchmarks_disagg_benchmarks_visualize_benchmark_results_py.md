# File: `benchmarks/disagg_benchmarks/visualize_benchmark_results.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 47 |
| Imports | json, matplotlib, pandas |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Visualize disaggregation benchmark results

**Mechanism:** Reads JSON results for disagg_prefill vs chunked_prefill across QPS levels (2,4,6,8). Generates matplotlib line plots comparing TTFT and ITL metrics (mean, median, p99) using pandas DataFrames. Saves plots to results/ directory with bmh style.

**Significance:** Analysis tool for disaggregation architecture evaluation. Visualizes latency characteristics of prefill/decode separation versus monolithic chunked prefill, helping identify performance trade-offs and optimal configurations.
