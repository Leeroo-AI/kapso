# Implementation: FastQwen3MoeModel

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
|-
! Domains
| [[domain::Models]], [[domain::MoE]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2026-01-09 15:46 GMT]]
|}

== Overview ==
Optimized patching module for Qwen 3 Mixture-of-Experts (MoE) models.

=== Description ===
This module provides Unsloth-optimized implementations for Qwen 3 MoE variants. It extends the base Qwen 3 optimizations with support for the MoE architecture including optimized expert routing and grouped GEMM operations.

=== Usage ===
Used automatically when loading Qwen 3 MoE models through `FastLanguageModel.from_pretrained()`.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth Unslothai_Unsloth]
* '''File:''' [https://github.com/unslothai/unsloth/blob/main/unsloth/models/qwen3_moe.py unsloth/models/qwen3_moe.py]
* '''Lines:''' 1-243

=== Import ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3-MoE-15B",
    max_seq_length=4096,
    load_in_4bit=True,
)
</syntaxhighlight>

== Qwen 3 MoE Architecture Notes ==

{| class="wikitable"
|-
! Feature !! Description
|-
| Mixture of Experts || Multiple expert FFN layers per block
|-
| Top-K Routing || Selects top experts per token
|-
| Shared Expert || Some variants include shared experts
|-
| QK Normalization || RMSNorm on Q and K before attention
|}

== Related Pages ==
* [[requires_env::Environment:Unslothai_Unsloth_CUDA_GPU_Environment]]
