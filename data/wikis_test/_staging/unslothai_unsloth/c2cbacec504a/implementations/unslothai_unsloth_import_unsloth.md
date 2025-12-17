# Implementation: unslothai_unsloth_import_unsloth

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|Unsloth Docs|https://docs.unsloth.ai]]
|-
! Domains
| [[domain::NLP]], [[domain::Optimization]], [[domain::Training]]
|-
! Last Updated
| [[last_updated::2025-12-17 15:00 GMT]]
|}

== Overview ==

Pattern for importing Unsloth before other ML libraries to enable automatic optimization patches for transformers, TRL, and PEFT.

=== Description ===

The `import unsloth` statement must be the **first import** in any training script. Unsloth modifies these libraries at import time to apply optimizations:

* **transformers**: Patches attention mechanisms, gradient checkpointing, RoPE embeddings
* **TRL**: Patches SFTTrainer, DPOTrainer, GRPOTrainer for memory efficiency
* **PEFT**: Patches LoRA operations with optimized Triton kernels

If these libraries are imported before Unsloth, the original (slower, more memory-intensive) implementations will be used instead.

=== Usage ===

Use this pattern at the **very beginning** of every Unsloth training script. The import triggers:
1. Library patching for transformers, TRL, PEFT
2. CUDA environment validation
3. bitsandbytes linking verification
4. Triton kernel registration

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unsloth]
* '''File:''' unsloth/__init__.py
* '''Lines:''' L1-287

=== Import Pattern ===
<syntaxhighlight lang="python">
# CRITICAL: Import unsloth FIRST, before any other ML libraries
import unsloth

# Now safe to import other libraries
from transformers import AutoTokenizer
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
</syntaxhighlight>

=== What Gets Patched ===
<syntaxhighlight lang="python">
# From unsloth/__init__.py - these patches are applied automatically:

# 1. Fix compatibility issues
fix_message_factory_issue()
fix_xformers_performance_issue()
patch_datasets()
patch_enable_input_require_grads()

# 2. Import optimized modules
from .models import *        # FastLanguageModel, FastVisionModel
from .save import *          # save_pretrained_merged, save_pretrained_gguf
from .chat_templates import * # get_chat_template
from .trainer import *       # UnslothTrainer patches

# 3. Patch TRL trainers
_patch_trl_trainer()
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| N/A || N/A || N/A || Import statement takes no arguments
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| FastLanguageModel || class || Optimized model loading class
|-
| FastVisionModel || class || Optimized vision model loading class
|-
| get_chat_template || function || Chat template configuration utility
|-
| Patched libraries || side effect || transformers, TRL, PEFT now use optimized code paths
|}

== Usage Examples ==

=== Basic Import Pattern ===
<syntaxhighlight lang="python">
# CORRECT: Unsloth first
import unsloth
from unsloth import FastLanguageModel

from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset
</syntaxhighlight>

=== Handling Import Warnings ===
<syntaxhighlight lang="python">
# If you see this warning, your imports are in wrong order:
# "WARNING: Unsloth should be imported before [trl, transformers, peft]"

# Fix by restructuring imports with unsloth at the top
import unsloth  # MUST be first
# ... rest of imports
</syntaxhighlight>

=== Environment Variables ===
<syntaxhighlight lang="python">
import os

# Optional: Disable auto padding-free optimization
os.environ["UNSLOTH_DISABLE_AUTO_PADDING_FREE"] = "1"

# Optional: Enable verbose logging
os.environ["UNSLOTH_ENABLE_LOGGING"] = "1"

# Then import unsloth
import unsloth
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:unslothai_unsloth_Environment_Initialization]]

=== Requires Environment ===
* [[requires_env::Environment:unslothai_unsloth_CUDA]]
