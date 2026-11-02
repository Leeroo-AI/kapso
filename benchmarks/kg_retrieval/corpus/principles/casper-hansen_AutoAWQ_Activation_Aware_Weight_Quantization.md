# Principle: Activation_Aware_Weight_Quantization

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration|https://arxiv.org/abs/2306.00978]]
* [[source::Repo|AutoAWQ|https://github.com/casper-hansen/AutoAWQ]]
|-
! Domains
| [[domain::LLMs]], [[domain::Quantization]]
|-
! Last Updated
| [[last_updated::2026-02-10 12:00 GMT]]
|}

== Overview ==

Weight-only compression that finds a small set of salient channels from activation statistics and rescales them before rounding, reducing error without backpropagation or layer reconstruction.

=== Description ===

A tiny fraction of weight channels, those multiplied by large activations, dominate output error when rounded. Scaling these channels up before rounding, and folding the inverse scale into the preceding operation, lowers their relative error. The scales are found by a small grid search on sample inputs, so the procedure is fast and relies little on the calibration data.

=== Usage ===

Choose it for 4-bit deployment when quality across domains and quantization speed both matter.

== Theoretical Basis ==

For a channel with input activation magnitude s_x, scaling its weights by s > 1 shrinks the relative rounding error by about 1/s, while the activation is divided by s. The scale is chosen as s = s_x^alpha with alpha searched in [0, 1] to minimise output error.

== Related Pages ==

* [[implemented_by::Implementation:casper-hansen_AutoAWQ_Quantize]]
