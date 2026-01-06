# File: `benchmarks/multi_turn/bench_dataset.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 600 |
| Classes | `Distribution`, `UniformDistribution`, `ConstantDistribution`, `ZipfDistribution`, `PoissonDistribution`, `LognormalDistribution`, `GenConvArgs` |
| Functions | `verify_field_exists`, `get_random_distribution`, `parse_input_json_file`, `print_conv_stats`, `generate_conversations`, `conversations_list_to_dict`, `conversations_dict_to_list` |
| Imports | abc, bench_utils, numpy, pandas, statistics, tqdm, transformers, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Dataset utilities for multi-turn conversation benchmarking.

**Mechanism:** Loads and processes multi-turn conversation datasets for benchmarking conversational inference performance.

**Significance:** Multi-turn benchmarks reflect real chatbot workloads with context carryover. Important for evaluating KV cache management.
