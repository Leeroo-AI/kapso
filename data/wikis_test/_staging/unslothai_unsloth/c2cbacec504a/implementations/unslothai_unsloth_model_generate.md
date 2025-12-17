# Implementation: unslothai_unsloth_model_generate

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|HuggingFace Generation|https://huggingface.co/docs/transformers/generation_strategies]]
|-
! Domains
| [[domain::NLP]], [[domain::Inference]], [[domain::Validation]]
|-
! Last Updated
| [[last_updated::2025-12-17 15:00 GMT]]
|}

== Overview ==

Concrete tool for generating text with trained models to verify training quality before export.

=== Description ===

`model.generate()` runs inference to produce text completions. After training, this is used to verify model quality before committing to export.

=== Usage ===

Use before model export to validate the model responds correctly to test prompts.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unsloth]
* '''File:''' unsloth/models/llama.py
* '''Lines:''' L2500-2550

=== Usage Example ===
<syntaxhighlight lang="python">
# After training, verify model quality
from unsloth import FastLanguageModel

FastLanguageModel.for_inference(model)

inputs = tokenizer("What is 2+2?", return_tensors="pt").to("cuda")
outputs = model.generate(
    **inputs,
    max_new_tokens = 64,
    temperature = 0.7,
)
print(tokenizer.decode(outputs[0]))
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:unslothai_unsloth_Training_Verification]]

=== Requires Environment ===
* [[requires_env::Environment:unslothai_unsloth_CUDA]]
