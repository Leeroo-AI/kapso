# Implementation: FastGemma2Model

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Paper|Gemma2|https://arxiv.org/abs/2408.00118]]
|-
! Domains
| [[domain::Models]], [[domain::NLP]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2026-01-09 15:46 GMT]]
|}

== Overview ==
Optimized patching module for Google Gemma 2 models with logit softcapping and sliding window attention support.

=== Description ===
This module provides Unsloth-optimized implementations for Gemma 2 models (9B, 27B). It includes support for logit softcapping in attention (a Gemma2 innovation), sliding window attention, and optimized inference with GEGLU activations.

=== Usage ===
Used automatically when loading Gemma 2 models through `FastLanguageModel.from_pretrained()`.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth Unslothai_Unsloth]
* '''File:''' [https://github.com/unslothai/unsloth/blob/main/unsloth/models/gemma2.py unsloth/models/gemma2.py]
* '''Lines:''' 1-654

=== Import ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="google/gemma-2-9b",
    max_seq_length=4096,
    load_in_4bit=True,
)
</syntaxhighlight>

== Gemma 2 Architecture Notes ==

{| class="wikitable"
|-
! Feature !! Description
|-
| Logit Softcapping || Soft cap on attention logits: t * tanh(logits / t)
|-
| Sliding Window || Alternating global and local attention
|-
| Query Pre-Attention Scaling || Uses query_pre_attn_scalar from config
|-
| GEGLU Activation || GELU-gated linear unit in MLP
|}

== Related Pages ==
* [[requires_env::Environment:Unslothai_Unsloth_CUDA_GPU_Environment]]
