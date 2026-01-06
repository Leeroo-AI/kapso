# File: `benchmarks/disagg_benchmarks/visualize_benchmark_results.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 47 |
| Imports | json, matplotlib, pandas |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Visualizes benchmark comparison between disagg_prefill and chunked_prefill architectures.

**Mechanism:** Loads JSON results from results/ directory for both disagg_prefill and chunked_prefill across different QPS levels (2, 4, 6, 8). Creates pandas DataFrames and generates line plots comparing six metrics: mean/median/p99 TTFT (time to first token) and ITL (inter-token latency). Uses matplotlib with 'bmh' style, plots both architectures with markers, and saves PNG figures for each metric. X-axis is QPS, Y-axis is the metric value (ms).

**Significance:** Analysis tool for comparing disaggregated vs. chunked prefill serving architectures. Visualizations help identify performance tradeoffs at different load levels. Disaggregated prefill may show better TTFT due to dedicated prefill resources, while chunked prefill might have simpler coordination. Critical for making architecture decisions based on empirical performance data. Results inform whether disaggregation benefits justify the added system complexity.
