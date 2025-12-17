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

**Purpose:** Generates synthetic multi-turn conversation datasets

**Mechanism:** Comprehensive dataset generator for multi-turn conversation benchmarking. Supports multiple statistical distributions (Uniform, Constant, Zipf, Poisson, Lognormal) for controlling conversation characteristics like num_turns, num_requests_per_turn, think_time, and token lengths. GenConvArgs class configures generation parameters. Parses input JSON datasets and generates synthetic conversations matching specified distribution patterns. Uses transformers tokenizer for token counting. Outputs conversation statistics (mean, median, percentiles) for validation. Converts between list and dict conversation formats.

**Significance:** Critical infrastructure for realistic multi-turn conversation benchmarking. Enables controlled testing of stateful serving scenarios with varying conversation patterns. Essential for evaluating prefix caching, KV cache management, and scheduling under multi-turn workloads. Helps simulate production conversation patterns (e.g., Zipf distribution for turn counts, Poisson for request timing). Key tool for stress testing conversational AI serving.
