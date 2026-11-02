# Implementation: BitsAndBytes_4bit_Config

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|bitsandbytes integration|https://huggingface.co/docs/transformers/quantization/bitsandbytes]]
|-
! Domains
| [[domain::LLMs]], [[domain::Quantization]]
|-
! Last Updated
| [[last_updated::2026-02-10 12:00 GMT]]
|}

== Overview ==

BitsAndBytesConfig in transformers selects 4-bit loading through the bitsandbytes library, with options for the data type, double quantization and the compute dtype of matrix products.

=== Description ===

Passing the config to from_pretrained makes the library replace linear layers with 4-bit versions while the checkpoint is read, so the model is never materialised at 16-bit. The stored weights are compressed block by block, and each block is expanded just in time for a product.

=== Usage ===

Pass it as quantization_config when loading a base model that will be trained with adapters or run for inference with a small memory footprint.

== Code Reference ==

<syntaxhighlight lang="python">
transformers.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_compute_dtype=torch.bfloat16)
</syntaxhighlight>

* ''bnb_4bit_quant_type'': "nf4" or "fp4"
* ''bnb_4bit_compute_dtype'': dtype used for the arithmetic after expansion

== Usage Examples ==

<syntaxhighlight lang="python">
model = AutoModelForCausalLM.from_pretrained(name, quantization_config=bnb_config,
                                             device_map="auto")
</syntaxhighlight>

== Related Pages ==

* [[requires_env::Environment:huggingface_transformers_BitsAndBytes_CUDA_Training]]
