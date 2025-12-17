# File: `src/peft/tuners/ia3/model.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 315 |
| Classes | `IA3Model` |
| Imports | __future__, dataclasses, layer, peft, re, torch, transformers, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Main IA3 model class managing adapter injection and weighted combination

**Mechanism:** IA3Model extends BaseTuner, creates appropriate layer types (Linear/Conv2d/Conv3d/Linear8bitLt/Linear4bit) based on base layer and quantization. _check_target_module_feedforward() determines per-module treatment via regex/string matching. _prepare_adapter_config() auto-fills target/feedforward modules from model-specific mappings. add_weighted_adapter() combines multiple adapters by weighted summation of ia3_l vectors

**Significance:** Orchestrates IA3 fine-tuning workflow - handles adapter injection with feedforward/attention distinction, supports multi-adapter composition, enables efficient model adaptation with <0.01% trainable parameters while maintaining near full fine-tuning performance
