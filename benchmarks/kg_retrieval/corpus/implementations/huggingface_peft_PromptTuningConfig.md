# Implementation: PromptTuningConfig

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

Configuration dataclass for prompt tuning in PEFT; num_virtual_tokens sets the length of the learned prefix and prompt_tuning_init chooses random or text-based starting values.

=== Description ===

The object tells get_peft_model to prepend a trainable embedding matrix to every input. With text initialisation, the embeddings begin as those of a supplied phrase, so training starts from a meaningful instruction.

=== Usage ===

Build it when adapting a frozen model by learning input embeddings only.

== Code Reference ==

<syntaxhighlight lang="python">
peft.PromptTuningConfig(task_type="CAUSAL_LM", num_virtual_tokens=20,
                        prompt_tuning_init="TEXT",
                        prompt_tuning_init_text="Classify the sentiment:",
                        tokenizer_name_or_path=name)
</syntaxhighlight>
