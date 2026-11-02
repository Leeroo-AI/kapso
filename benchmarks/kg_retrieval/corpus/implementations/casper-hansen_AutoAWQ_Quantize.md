# Implementation: Quantize

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|AutoAWQ|https://github.com/casper-hansen/AutoAWQ]]
* [[source::Doc|AutoAWQ documentation|https://casper-hansen.github.io/AutoAWQ/]]
|-
! Domains
| [[domain::LLMs]], [[domain::Quantization]]
|-
! Last Updated
| [[last_updated::2026-02-10 12:00 GMT]]
|}

== Overview ==

AutoAWQForCausalLM is the entry class of the AutoAWQ library, and its quantize method compresses a loaded causal language model to 4-bit weights for fast inference with fused kernels. After loading a model and its tokenizer, the method pushes a small calibration dataset through the network one block at a time, collects activation statistics for every input channel, searches for per-channel scaling factors that protect the most salient weights, applies them, and then rounds the weights; the result is stored with save_quantized. The behaviour is set by a quant_config dictionary with the keys zero_point, q_group_size, w_bit and version, where w_bit gives the target bit width, q_group_size the number of weights sharing one scale, and zero_point whether asymmetric quantization is used, while version picks the GEMM or GEMV kernel layout.

=== Description ===

AutoAWQForCausalLM is the main class of the AutoAWQ library, which compresses language-model weights to 4 bits for smaller, faster inference using the AWQ algorithm. Its quantize method turns a loaded model into a compressed one and save_quantized writes the result. The AutoAWQ project is deprecated and its repository archived, with its development continuing in the vLLM project's llm-compressor.

=== Usage ===

Load the model, call quantize with a configuration and optionally a calibration dataset, then save.

== Code Reference ==

<syntaxhighlight lang="python">
from awq import AutoAWQForCausalLM
quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}
model = AutoAWQForCausalLM.from_pretrained(name)
model.quantize(tokenizer, quant_config=quant_config)
model.save_quantized(out_dir)
</syntaxhighlight>
