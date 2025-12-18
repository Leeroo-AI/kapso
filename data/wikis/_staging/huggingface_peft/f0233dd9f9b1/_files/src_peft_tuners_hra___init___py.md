# File: `src/peft/tuners/hra/__init__.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 24 |
| Imports | config, layer, model, peft |

## Understanding

**Status:** ✅ Explored

**Purpose:** Package initialization file for the HRA (Householder Reflection Adaptation) PEFT method that exports key components and registers HRA as a PEFT method in the library.

**Mechanism:** The file imports HRAConfig, HRAConv2d, HRALayer, HRALinear, and HRAModel from their respective modules, exposes them via __all__, and calls register_peft_method() to register HRA with the PEFT framework, mapping the name "hra" to HRAConfig and HRAModel. This registration enables users to instantiate HRA adapters using the standard PEFT API.

**Significance:** This is a core initialization file that makes HRA available as a first-class PEFT method. HRA (https://huggingface.co/papers/2405.17484) uses Householder reflections to create orthogonal transformations of pretrained weights, offering a geometrically-motivated alternative to low-rank methods. It is particularly effective for vision models (Stable Diffusion, ViT) and can be applied to both Linear and Conv2d layers. The registration here integrates it into PEFT's adapter ecosystem.
