# File: `examples/offline_inference/torchrun_example.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 76 |
| Imports | torch, vllm |

## Understanding

**Status:** ✅ Explored

**Purpose:** Demonstrates tensor/pipeline parallelism with torchrun

**Mechanism:** Uses torchrun with external_launcher backend for TP=2, PP=2 configuration. All ranks run same code but only rank 0 prints outputs. Shows explicit seed setting for deterministic cross-rank results. Includes tips for distributed communication patterns.

**Significance:** Example showing torchrun-based tensor and pipeline parallelism with external process management.
