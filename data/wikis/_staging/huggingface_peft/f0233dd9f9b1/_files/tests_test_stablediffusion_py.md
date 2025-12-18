# File: `tests/test_stablediffusion.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 387 |
| Classes | `TestStableDiffusionModel` |
| Functions | `skip_if_not_lora` |
| Imports | copy, dataclasses, diffusers, numpy, peft, pytest, testing_common, testing_utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests for Stable Diffusion integration

**Mechanism:** Tests PEFT adapters (LoRA, LoHa, LoKr, OFT, HRA, BOFT) on Stable Diffusion pipelines with separate text_encoder and unet configs, including generation and low-level adapter injection

**Significance:** Test coverage for diffusers integration
