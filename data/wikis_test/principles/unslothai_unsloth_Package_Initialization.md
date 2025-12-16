{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|Installation Guide|https://docs.unsloth.ai/get-started/installation]]
|-
! Domains
| [[domain::Python]], [[domain::Installation]], [[domain::Configuration]]
|-
! Last Updated
| [[last_updated::2025-12-16 18:00 GMT]]
|}

== Overview ==

Initialization process that imports Unsloth and automatically patches HuggingFace libraries (transformers, TRL, PEFT) for optimized performance.

=== Description ===

Package initialization in Unsloth involves importing the library, which triggers automatic monkey-patching of underlying libraries:

**What Gets Patched:**
- transformers: Attention modules, loss functions, gradient checkpointing
- TRL: Trainer classes for RL workloads
- PEFT: LoRA operations with fused kernels
- bitsandbytes: Quantization operations

**Import Order Matters:**
Unsloth must be imported before other ML libraries to ensure patches are applied correctly.

**Automatic Optimizations:**
- Cross-entropy loss becomes chunked for large vocabularies
- Attention uses Flash Attention, xformers, or optimized SDPA
- RMS normalization uses custom Triton kernels

=== Usage ===

Always import Unsloth first in your training scripts, before importing other ML libraries.

== Practical Guide ==

=== Standard Import Order ===
<syntaxhighlight lang="python">
# 1. Import Unsloth FIRST (triggers automatic patching)
from unsloth import FastLanguageModel

# 2. Then import other libraries
import torch
from transformers import TrainingArguments
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
</syntaxhighlight>

=== RL Training Import Order ===
<syntaxhighlight lang="python">
# 1. Import and patch for RL
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("grpo", FastLanguageModel)

# 2. Then import TRL trainers
from trl import GRPOConfig, GRPOTrainer
</syntaxhighlight>

=== Verification ===
<syntaxhighlight lang="python">
import unsloth
print(f"Unsloth version: {unsloth.__version__}")

# Check if patches applied
import transformers
print(f"transformers patched: {hasattr(transformers, '_unsloth_patched')}")
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:unslothai_unsloth_FastLanguageModel]]

=== Used In Workflows ===
* [[used_by::Workflow:unslothai_unsloth_QLoRA_Finetuning]]
* [[used_by::Workflow:unslothai_unsloth_Vision_Language_Model_Finetuning]]
* [[used_by::Workflow:unslothai_unsloth_GRPO_Reinforcement_Learning]]
