# File: `tests/test_gptqmodel.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 563 |
| Classes | `PeftGPTQModelCommonTests`, `PeftGPTQModelTests` |
| Imports | accelerate, gc, os, peft, pytest, tempfile, testing_utils, torch, transformers, unittest |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests for PEFT adapters with GPTQ-quantized models on CPU and GPU.

**Mechanism:** Contains two test classes - PeftGPTQModelCommonTests for basic GPTQ+PEFT operations (loading, generation, multi-adapter support) and PeftGPTQModelTests for comprehensive training tests with LoRA, OFT, and AdaLoRA on quantized models. Tests single-GPU and multi-GPU training, adapter naming, and loading pre-trained GPTQ models with adapters. Validates that quantized models work correctly with kbit training preparation.

**Significance:** Essential for validating PEFT compatibility with GPTQ quantization, which enables efficient fine-tuning of large models with reduced memory footprint. Ensures users can train adapters on 4-bit quantized models, a critical use case for resource-constrained environments.
