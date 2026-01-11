# Implementation: FastGraniteModel

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
|-
! Domains
| [[domain::Models]], [[domain::NLP]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2026-01-09 15:46 GMT]]
|}

== Overview ==
Optimized patching module for IBM Granite models.

=== Description ===
This module provides Unsloth-optimized implementations for IBM Granite models. It includes optimized attention forward passes and efficient inference mode with memory management.

=== Usage ===
Used automatically when loading Granite models through `FastLanguageModel.from_pretrained()`.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth Unslothai_Unsloth]
* '''File:''' [https://github.com/unslothai/unsloth/blob/main/unsloth/models/granite.py unsloth/models/granite.py]
* '''Lines:''' 1-610

=== Import ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="ibm-granite/granite-3b-code-instruct",
    max_seq_length=4096,
    load_in_4bit=True,
)
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:Unslothai_Unsloth_CUDA_GPU_Environment]]
