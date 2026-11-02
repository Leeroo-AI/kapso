# Implementation: LoraConfig

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Doc|PEFT documentation|https://huggingface.co/docs/peft]]
|-
! Domains
| [[domain::LLMs]], [[domain::Fine_Tuning]], [[domain::PEFT]]
|-
! Last Updated
| [[last_updated::2026-02-10 12:00 GMT]]
|}

== Overview ==

LoraConfig is the configuration dataclass in the PEFT library that holds every setting of a LoRA adapter. It is passed to get_peft_model, which wraps a pretrained transformers model and injects the trainable low-rank matrices; the same object is serialised next to the saved adapter so that it can be reloaded later. Its central fields are r, the rank of the update matrices, and lora_alpha, which scales the update. The target_modules field lists the module names, or a regular expression, that receive adapters; when it is omitted, PEFT falls back to a default list for known architectures. Setting use_dora=True switches the adapter to the weight-decomposed DoRA variant, while lora_dropout, bias, task_type and modules_to_save control regularisation and which extra layers stay trainable.

=== Description ===

LoraConfig is the PEFT configuration class for LoRA, a parameter-efficient fine-tuning method that trains small low-rank matrices in place of the full weights of a pretrained transformers model. It gathers the settings of one adapter and is consumed by get_peft_model.

=== Usage ===

Create one configuration per adapter and pass it together with the base model.

== Code Reference ==

<syntaxhighlight lang="python">
peft.LoraConfig(r=8, lora_alpha=8, target_modules=None, lora_dropout=0.0,
                bias="none", task_type=None, use_dora=False, modules_to_save=None)
</syntaxhighlight>

== Usage Examples ==

<syntaxhighlight lang="python">
from peft import LoraConfig, get_peft_model

config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
                    lora_dropout=0.05, task_type="CAUSAL_LM")
model = get_peft_model(base_model, config)
model.print_trainable_parameters()
</syntaxhighlight>
