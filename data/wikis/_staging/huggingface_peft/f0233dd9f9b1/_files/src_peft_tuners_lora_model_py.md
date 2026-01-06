# File: `src/peft/tuners/lora/model.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 872 |
| Classes | `LoraModel` |
| Imports | __future__, aqlm, awq, config, contextlib, dataclasses, eetq, functools, gptq, hqq, ... +11 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** LoRA model orchestration

**Mechanism:** Implements LoraModel class (inheriting from BaseTuner) that orchestrates applying LoRA adapters to base models. Handles module injection by replacing target layers with LoRA-enhanced versions using dispatch functions for different quantization backends (GPTQ, AWQ, BNB, AQLM, EETQ, HQQ, TorchAO, INC) and tensor parallelism (Megatron). Manages adapter lifecycle: creation, activation, merging, unmerging, and state dict operations. Supports special initializations (PiSSA, OLoRA, LoftQ, CorDA, Eva) and various LoRA variants (DoRA, Arrow). Includes methods for scaling adapters, managing multiple adapters, and handling quantized models.

**Significance:** The central orchestrator for LoRA fine-tuning in PEFT. Provides the high-level API that users interact with to add adapters to models, manages complexity of different hardware backends and quantization formats, and ensures adapters work correctly across diverse model architectures. Essential for making LoRA practical and accessible.
