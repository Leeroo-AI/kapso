# File: `examples/offline_inference/mlpspeculator.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 72 |
| Functions | `time_generation`, `main` |
| Imports | gc, time, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Demonstrates MLPSpeculator for speculative decoding using a trained MLP head to predict future tokens and speed up generation.

**Mechanism:** Configures LLM with speculative_config using MLPSpeculator (trained predictor head) to generate draft tokens. Compares generation speed with and without speculation, showing latency improvements. The MLP head predicts multiple tokens ahead, which are verified by the main model in parallel.

**Significance:** Shows advanced speculative decoding technique using learned predictors rather than separate draft models. Important optimization for reducing generation latency through parallel token verification.
