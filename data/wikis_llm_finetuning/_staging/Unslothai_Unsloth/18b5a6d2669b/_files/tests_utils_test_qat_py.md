# File: `tests/utils/test_qat.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 156 |
| Classes | `_CountingFakeQuantizer` |
| Functions | `test_lora_model_fake_quantize` |
| Imports | pytest, torch, torchao, typing, unsloth |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests for Quantization-Aware Training (QAT) integration, verifying that linear layers are correctly wrapped with fake quantizers for both full fine-tuning and LoRA training.

**Mechanism:** Tests load the Qwen3-1.7B model with QAT schemes (fp8-int4 or fp8-fp8) and verify that linear layers in attention and MLP blocks are wrapped with FakeQuantizedLinear modules containing appropriate fake quantizers (Float8FakeQuantizer for activations, Int4WeightPreshuffledFakeQuantizer or Float8FakeQuantizer for weights). Uses a _CountingFakeQuantizer to verify quantizers are actually invoked during forward passes, checking that activation quantization happens once per block (at q_proj/gate_proj for LoRA) while weight quantization happens for all layers.

**Significance:** QAT enables training models that maintain accuracy when deployed with low-precision inference (FP8/INT4), which dramatically reduces memory usage and increases throughput. These tests ensure Unsloth's QAT integration with torchao works correctly for both full fine-tuning and parameter-efficient LoRA training, validating that quantization simulation happens during training to prepare models for quantized deployment.
