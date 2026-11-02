# Workflow: Quantize_And_Deploy_A_Model

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Optimum|https://github.com/huggingface/optimum]]
* [[source::Paper|GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers|https://arxiv.org/abs/2210.17323]]
* [[source::Paper|AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration|https://arxiv.org/abs/2306.00978]]
|-
! Domains
| [[domain::LLMs]], [[domain::Quantization]], [[domain::Deployment]]
|-
! Last Updated
| [[last_updated::2026-02-10 12:00 GMT]]
|}

== Overview ==

Recipe for shrinking a trained language model to low-bit weights with a calibration pass, checking that quality holds, and exporting it for an inference server.

=== Description ===

Weight-only compression lowers memory and often latency, but accuracy depends on the method, the bit width and the data used to calibrate it. The recipe measures a baseline, compresses, re-measures, and only then packages the artifact.

=== Usage ===

Follow it when a model must run on hardware smaller than its original precision requires.

== Execution Steps ==

=== Step 1: Baseline ===

Measure perplexity and one or two task metrics of the original model on held-out text.

=== Step 2: Choose a method and bit width ===

Pick a post-training method and a target of 4 bits per weight with a group size; groups of 128 are a common compromise between accuracy and size.

=== Step 3: Compress with representative inputs ===

Run a few hundred token sequences drawn from the target domain through the method so it can measure the activations it must protect.

=== Step 4: Re-evaluate ===

Repeat the baseline measurements on the compressed model and accept it only if the degradation is within the budget.

=== Step 5: Package and publish an endpoint ===

Save the packed weights with their configuration and start an inference server that supports the chosen format.

== GitHub URL ==

[https://github.com/huggingface/optimum huggingface/optimum]
