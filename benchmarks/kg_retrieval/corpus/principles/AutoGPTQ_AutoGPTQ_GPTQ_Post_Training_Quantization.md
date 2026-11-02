# Principle: GPTQ_Post_Training_Quantization

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers|https://arxiv.org/abs/2210.17323]]
* [[source::Repo|AutoGPTQ|https://github.com/AutoGPTQ/AutoGPTQ]]
|-
! Domains
| [[domain::LLMs]], [[domain::Quantization]]
|-
! Last Updated
| [[last_updated::2026-02-10 12:00 GMT]]
|}

== Overview ==

Post-training method that rounds weights layer by layer to a low bit width, using second-order information from sample inputs to update the remaining weights and offset each rounding error.

=== Description ===

For every linear layer, GPTQ minimises the squared error between the layer's original output and its quantized output over a small set of inputs. Weights are rounded one column at a time, and the error introduced by each is spread over the not-yet-rounded columns using the inverse of the layer's Hessian. No gradient-based retraining is needed.

=== Usage ===

Use it to produce 3 or 4 bit weights for generative models when some calibration text is available and a one-off compression time of minutes to hours is acceptable.

== Theoretical Basis ==

For a layer with weights W and inputs X, the objective is argmin ||WX - W'X||^2 over quantized W'. The Hessian is H = 2XX^T. After quantizing column q, the remaining weights receive the update delta = -(w_q - quant(w_q)) / [H^-1]_qq * (H^-1)_{:,q}, with the inverse obtained once by a Cholesky factorisation.

== Related Pages ==

* [[implemented_by::Implementation:AutoGPTQ_AutoGPTQ_Quantize]]
* [[uses_heuristic::Heuristic:AutoGPTQ_AutoGPTQ_Calibration_Set_Size]]
