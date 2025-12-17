# Heuristic: unslothai_unsloth_Import_Order

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Discussion|GitHub Issues|https://github.com/unslothai/unsloth/issues]]
|-
! Domains
| [[domain::Configuration]], [[domain::Optimization]], [[domain::Python]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

## Overview

Critical initialization heuristic: **Import `unsloth` BEFORE `transformers`, `trl`, or `peft`** to ensure all performance optimizations are applied.

### Description

Unsloth optimizes PyTorch and HuggingFace libraries by monkey-patching their internal code at import time. These patches include:
- Triton-based kernels for RoPE, LayerNorm, and Cross-Entropy Loss
- Memory-efficient gradient checkpointing
- Optimized bitsandbytes integration
- TRL trainer backwards compatibility patches

If the target libraries are imported before Unsloth, the original (slower, more memory-intensive) implementations will be used instead, potentially causing OOM errors or 2-5x slower training.

### Usage

Apply this heuristic **always** when using Unsloth. This is the very first step in any Unsloth workflow.

**Symptoms of incorrect import order:**
- Training is slower than expected
- OOM errors that shouldn't occur given VRAM
- Warning message: "Unsloth should be imported before [trl, transformers, peft]"

## The Insight (Rule of Thumb)

* **Action:** Place `import unsloth` as the **first import** in your script, before any HuggingFace libraries.
* **Value:** N/A (import ordering, not a numeric setting)
* **Trade-off:** None - this is purely beneficial

**Correct Order:**
<syntaxhighlight lang="python">
# CORRECT - Unsloth first
import unsloth
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer
</syntaxhighlight>

**Incorrect Order (causes warning and slower training):**
<syntaxhighlight lang="python">
# WRONG - transformers/trl imported before unsloth
from transformers import AutoModelForCausalLM
from trl import SFTTrainer
import unsloth  # TOO LATE! Optimizations won't apply
</syntaxhighlight>

## Reasoning

Unsloth modifies the internals of transformers, trl, and peft at import time. Python's module system caches imported modules in `sys.modules`. Once a module is cached:

1. Subsequent imports return the cached version
2. Monkey-patches applied later don't affect the already-imported code
3. Any references to the original (unpatched) functions remain unoptimized

The code checks for pre-imported critical modules and warns users:

<syntaxhighlight lang="python">
# From __init__.py:24-57
critical_modules = ["trl", "transformers", "peft"]
already_imported = [mod for mod in critical_modules if mod in sys.modules]

if already_imported:
    warnings.warn(
        f"WARNING: Unsloth should be imported before [{', '.join(already_imported)}] "
        f"to ensure all optimizations are applied. Your code may run slower or encounter "
        f"memory issues without these optimizations.\n\n"
        f"Please restructure your imports with 'import unsloth' at the top of your file.",
        stacklevel = 2,
    )
</syntaxhighlight>

## Related Pages

* [[uses_heuristic::Implementation:unslothai_unsloth_import_unsloth]]
* [[uses_heuristic::Workflow:unslothai_unsloth_QLoRA_Finetuning]]
* [[uses_heuristic::Workflow:unslothai_unsloth_GRPO_Reinforcement_Learning]]
