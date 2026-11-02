# Implementation: Quantize

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|AutoGPTQ|https://github.com/AutoGPTQ/AutoGPTQ]]
* [[source::Doc|AutoGPTQ README|https://github.com/AutoGPTQ/AutoGPTQ#readme]]
|-
! Domains
| [[domain::LLMs]], [[domain::Quantization]]
|-
! Last Updated
| [[last_updated::2026-02-10 12:00 GMT]]
|}

== Overview ==

AutoGPTQForCausalLM is the main class of the AutoGPTQ library, and its quantize method is the call that compresses a loaded causal language model to a low bit width. The model is first loaded together with a quantization configuration object; quantize then processes the transformer blocks one after another, solving a per-layer reconstruction problem against the outputs recorded for sample inputs, and finally save_quantized writes the packed weights to disk for later loading. The call takes a list of tokenised calibration examples, each holding input_ids and an attention_mask, and its behaviour is governed by the BaseQuantizeConfig fields bits, group_size and desc_act, the last of which processes weight columns in order of decreasing activation magnitude so that the largest are rounded first.

=== Description ===

AutoGPTQForCausalLM is the main class of the AutoGPTQ library, which compresses language-model weights to a low bit width with the GPTQ algorithm. Its quantize method turns a loaded model into a compressed one, calibrated on sample inputs, and save_quantized writes the result. The AutoGPTQ repository is archived and unmaintained, and its README points new work to GPTQModel.

=== Usage ===

Prepare calibration examples from the target domain, call quantize once, and save the result.

== Code Reference ==

<syntaxhighlight lang="python">
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
config = BaseQuantizeConfig(bits=4, group_size=128, desc_act=False)
model = AutoGPTQForCausalLM.from_pretrained(name, config)
model.quantize(examples)
model.save_quantized(out_dir)
</syntaxhighlight>

== Related Pages ==

* [[uses_heuristic::Heuristic:AutoGPTQ_AutoGPTQ_Calibration_Set_Size]]
