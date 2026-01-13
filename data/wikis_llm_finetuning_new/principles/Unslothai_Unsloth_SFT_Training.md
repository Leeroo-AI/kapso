# Principle: SFT_Training

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|TRL SFTTrainer|https://huggingface.co/docs/trl/sft_trainer]]
* [[source::Paper|Fine-Tuning Language Models from Human Preferences|https://arxiv.org/abs/1909.08593]]
* [[source::Paper|QLoRA|https://arxiv.org/abs/2305.14314]]
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
|-
! Domains
| [[domain::NLP]], [[domain::Deep_Learning]], [[domain::Training]]
|-
! Last Updated
| [[last_updated::2026-01-12 00:00 GMT]]
|}

== Overview ==

Mechanism for supervised fine-tuning (SFT) of language models using next-token prediction on instruction-response pairs.

=== Description ===

Supervised Fine-Tuning (SFT) trains a language model to generate appropriate responses to given prompts by minimizing the cross-entropy loss on human-written examples. The model learns to predict each token in the target response given the prompt and preceding response tokens.

Key aspects of SFT:
* **Causal Language Modeling**: Model predicts next token autoregressively
* **Teacher Forcing**: Ground truth tokens provided during training
* **Response-Only Loss**: Optionally mask prompt tokens from loss computation
* **Sequence Packing**: Combine multiple short sequences for efficiency

SFT is the foundation for instruction-following capabilities and often precedes RLHF or DPO alignment stages.

=== Usage ===

Use this principle when:
* Teaching a model to follow instructions in a specific format
* Fine-tuning on conversational or QA datasets
* Creating a base for further alignment training (RLHF, DPO, GRPO)
* The warmup stage of reinforcement learning workflows

This is the core training step in QLoRA fine-tuning, executed after all configuration is complete.

== Theoretical Basis ==

The SFT objective minimizes cross-entropy loss over response tokens:

<math>
\mathcal{L}_{SFT} = -\sum_{t=1}^{T} \log P(y_t | x, y_{<t}; \theta)
</math>

Where:
- x is the prompt/input sequence
- y = (y_1, ..., y_T) is the target response
- θ are the model parameters (LoRA weights in QLoRA)

'''Response-Only Training:'''
When using `train_on_responses_only`, the loss is masked for prompt tokens:

<syntaxhighlight lang="python">
# Pseudo-code for response-only loss
for position, token in enumerate(sequence):
    if is_prompt_token[position]:
        loss_mask[position] = 0  # Ignore prompt in loss
    else:
        loss_mask[position] = 1  # Include response in loss

loss = cross_entropy(logits, labels) * loss_mask
</syntaxhighlight>

'''Sequence Packing:'''
Multiple short sequences packed into one for efficiency:
<syntaxhighlight lang="text">
[BOS] Sample1 [EOS] [BOS] Sample2 [EOS] [BOS] Sample3 [EOS] [PAD]...
</syntaxhighlight>

Attention masks ensure samples don't attend to each other.

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Unslothai_Unsloth_SFTTrainer_train]]

