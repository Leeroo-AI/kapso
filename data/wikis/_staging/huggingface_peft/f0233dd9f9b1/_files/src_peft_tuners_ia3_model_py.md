# File: `src/peft/tuners/ia3/model.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 315 |
| Classes | `IA3Model` |
| Imports | __future__, dataclasses, layer, peft, re, torch, transformers, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements IA3Model, the main adapter model class that applies IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations) to pretrained transformers by replacing target layers with IA3-enabled versions that learn per-dimension scaling vectors.

**Mechanism:** IA3Model extends BaseTuner and provides comprehensive layer replacement logic. The _create_new_module() method handles various layer types: Linear, Conv1D, Conv2d, Conv3d, and quantized variants (Linear8bitLt, Linear4bit), creating appropriate IA3-wrapped versions. It determines whether each module is feedforward by checking if its key matches feedforward_modules (using regex or exact/suffix matching via _check_target_module_exists). The _create_and_replace() method performs the actual replacement, handling both new and existing IA3 layers. The _check_target_module_exists() helper performs flexible module name matching (exact, suffix, or regex). The class maintains mappings for target modules and feedforward modules based on model architecture.

**Significance:** This is the core model class for IA3 (https://huggingface.co/papers/2205.05638), responsible for converting pretrained models into IA3-adapted versions. The key innovation is the feedforward/non-feedforward distinction: attention modules (K, Q, V) get output scaling while feedforward modules (FFN) get input scaling. This asymmetry is crucial for IA3's effectiveness. The comprehensive support for quantized layers (8-bit, 4-bit) makes IA3 compatible with memory-efficient inference. IA3's extreme parameter efficiency (only d scalars per layer) makes it ideal for scenarios requiring many task-specific adapters.
