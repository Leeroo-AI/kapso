{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Paper|QLoRA|https://arxiv.org/abs/2305.14314]]
* [[source::Doc|Transformers Quantization|https://huggingface.co/docs/transformers/main_classes/quantization]]
|-
! Domains
| [[domain::Model_Loading]], [[domain::Quantization]], [[domain::Memory_Optimization]]
|-
! Last Updated
| [[last_updated::2025-01-15 12:00 GMT]]
|}

== Overview ==

Principle for loading transformer models with 4-bit or 8-bit quantization using bitsandbytes, enabling large model training on consumer hardware.

=== Description ===

Quantized Model Loading loads a pretrained model with on-the-fly quantization applied. The model weights are converted to 4-bit (NF4) or 8-bit precision during loading, reducing memory footprint by approximately 4x compared to float16.

Key aspects:
* Weights are quantized during loading, not pre-quantized
* `device_map="auto"` is required for proper memory management
* Attention and linear layers use quantized weights
* Model is frozen after loading (training handled by adapters)

=== Usage ===

Apply after configuring BitsAndBytesConfig:
* Pass quantization_config to `from_pretrained()`
* Always use `device_map="auto"`
* Model will have Linear4bit/Linear8bitLt layers

== Theoretical Basis ==

'''NormalFloat4 (NF4) Quantization:'''

NF4 is information-theoretically optimal for normally distributed weights:
* Divides the range into 16 non-uniform bins
* Bins are optimized for Gaussian distribution
* Better preserves weight information than uniform quantization

'''Memory Savings:'''

| Precision | Bytes/Param | 7B Model Size |
|-----------|-------------|---------------|
| float32 | 4 | ~28 GB |
| float16 | 2 | ~14 GB |
| 4-bit | 0.5 | ~3.5 GB |

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_peft_AutoModel_from_pretrained_quantized]]
