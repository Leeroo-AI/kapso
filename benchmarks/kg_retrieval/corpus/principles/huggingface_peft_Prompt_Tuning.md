# Principle: Prompt_Tuning

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|The Power of Scale for Parameter-Efficient Prompt Tuning|https://arxiv.org/abs/2104.08691]]
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
|-
! Domains
| [[domain::LLMs]], [[domain::Fine_Tuning]], [[domain::PEFT]]
|-
! Last Updated
| [[last_updated::2026-02-10 12:00 GMT]]
|}

== Overview ==

Adaptation method that keeps every model weight frozen and learns a small matrix of continuous embeddings prepended to the input sequence.

=== Description ===

A fixed number of trainable vectors is placed in front of the token embeddings of each input. Only those vectors receive gradients, so a task is stored as a tiny tensor. Quality approaches full fine-tuning as the base model grows.

=== Usage ===

Use it when the number of tasks is large and per-task storage must be minimal, and the base model is big enough to respond well to it.

== Theoretical Basis ==

Given input embeddings X of shape (n, e), the model receives the concatenation [P; X] with P of shape (p, e) trainable. Initialising P from the embeddings of real words usually converges faster than random initialisation.

== Related Pages ==

* [[implemented_by::Implementation:huggingface_peft_PromptTuningConfig]]
