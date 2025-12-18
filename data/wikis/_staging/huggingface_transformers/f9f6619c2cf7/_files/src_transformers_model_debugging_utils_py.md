# File: `src/transformers/model_debugging_utils.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 456 |
| Functions | `prune_outputs_if_children`, `is_layer_block`, `prune_intermediate_layers`, `log_model_debug_trace`, `model_addition_debugger_context` |
| Imports | contextlib, functools, io, json, os, re, typing, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Developer tool for debugging model architectures by capturing and serializing all forward pass inputs/outputs into structured JSON traces.

**Mechanism:** The `model_addition_debugger_context()` context manager wraps every module's forward method in the model with debugging logic. During forward passes, it captures inputs (args, kwargs) and outputs for each module, building a hierarchical call tree that mirrors the model structure. Tensors are serialized either as repr() strings (use_repr=True) or saved to separate SafeTensors files (use_repr=False). The system includes utilities to prune intermediate layers for readability, sanitize memory addresses for stable diffs, handle DTensor representations in distributed settings, and only record on rank 0 during distributed execution. Outputs two JSON files: FULL_TENSORS.json (complete trace) and SUMMARY.json (without tensor values for easier navigation).

**Significance:** This is a specialized tool for transformer model contributors adding new architectures. It enables detailed debugging of model internals, comparison of different implementations, and verification of tensor shapes/values throughout the forward pass. Critical for catching subtle bugs in new model integrations and ensuring correctness across different frameworks.
