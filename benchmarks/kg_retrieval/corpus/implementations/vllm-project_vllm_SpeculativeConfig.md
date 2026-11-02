# Implementation: SpeculativeConfig

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Doc|vLLM documentation|https://docs.vllm.ai]]
|-
! Domains
| [[domain::LLMs]], [[domain::Inference]]
|-
! Last Updated
| [[last_updated::2026-02-10 12:00 GMT]]
|}

== Overview ==

Configuration object in vLLM that names the draft model or method used for speculative decoding and the number of tokens it may propose per step.

=== Description ===

It is supplied as the speculative_config argument of the engine. Its model field gives the draft checkpoint and num_speculative_tokens bounds the proposal length; alternative methods such as n-gram lookup need no separate model.

=== Usage ===

Provide it when creating the engine; leaving it unset disables the feature.

== Code Reference ==

<syntaxhighlight lang="python">
llm = LLM(model="target-model",
          speculative_config={"model": "draft-model", "num_speculative_tokens": 5})
</syntaxhighlight>
