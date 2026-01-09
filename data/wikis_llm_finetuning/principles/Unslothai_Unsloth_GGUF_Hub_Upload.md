# Principle: GGUF_Hub_Upload

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|HuggingFace Hub|https://huggingface.co/docs/huggingface_hub/guides/upload]]
|-
! Domains
| [[domain::Model_Sharing]], [[domain::HuggingFace_Hub]], [[domain::GGUF]]
|-
! Last Updated
| [[last_updated::2026-01-09 16:00 GMT]]
|}

== Overview ==

Uploading GGUF-quantized models to HuggingFace Hub for sharing and distribution.

=== Description ===

GGUF Hub Upload handles the full upload pipeline:

1. **Repository Creation**: Create or reuse HuggingFace repository
2. **GGUF Upload**: Upload quantized model files
3. **Metadata Upload**: Include config.json and Modelfile
4. **README Generation**: Create model card with usage instructions

=== Usage ===

Use after GGUF export to share quantized models publicly or privately.

== Theoretical Basis ==

=== Upload Strategy ===

For large GGUF files:
* Single file upload for files < 50GB
* Automatic chunking handled by huggingface_hub
* Progress tracking during upload

=== Repository Structure ===

Typical GGUF repository layout:
```
my-model-GGUF/
├── README.md
├── config.json
├── Modelfile
├── model-Q4_K_M.gguf
├── model-Q5_K_M.gguf
└── model-Q8_0.gguf
```

=== Naming Convention ===

GGUF files follow naming pattern:
```
{model_name}.{quantization}.gguf
```

Example: `llama-3-8b.Q4_K_M.gguf`

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Unslothai_Unsloth_push_to_hub_gguf]]

