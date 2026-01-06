{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Doc|Transformers Generation|https://huggingface.co/docs/transformers/main_classes/text_generation]]
|-
! Domains
| [[domain::Inference]], [[domain::Text_Generation]], [[domain::Model_Serving]]
|-
! Last Updated
| [[last_updated::2025-01-15 12:00 GMT]]
|}

== Overview ==

Principle for executing inference through a PEFT model, including text generation and forward passes with adapter-augmented behavior.

=== Description ===

Inference Execution covers running forward passes and generation through PEFT models. The adapter modifies the base model's behavior according to its learned weights. Common patterns include:
* Text generation using `model.generate()` for autoregressive models
* Classification via forward pass and logit extraction
* Embedding extraction from intermediate layers

The PEFT wrapper is transparent to inference - standard transformers methods work unchanged.

=== Usage ===

Apply this principle when running inference on a loaded PEFT model:
* Use `model.generate()` for text generation with sampling/beam search
* Use direct forward pass for classification or embedding tasks
* Configure generation parameters appropriate for your use case

== Theoretical Basis ==

'''Adapter-Augmented Forward Pass:'''

During inference, each adapted layer computes:
<math>h = W_0 x + B A x = (W_0 + BA) x</math>

Where:
* <math>W_0</math> is the frozen base model weight
* <math>B</math>, <math>A</math> are the learned LoRA matrices
* The adapter adds a task-specific modification to the base output

'''Generation Process:'''

For autoregressive generation:
1. Forward pass through adapted model
2. Sample/select next token from output logits
3. Append token and repeat

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_peft_model_generate]]
