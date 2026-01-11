# Principle: Vision_Model_Saving

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|HuggingFace Saving|https://huggingface.co/docs/transformers/main_classes/model]]
|-
! Domains
| [[domain::Computer_Vision]], [[domain::Model_Serialization]], [[domain::Multimodal]]
|-
! Last Updated
| [[last_updated::2026-01-09 16:00 GMT]]
|}

== Overview ==

Techniques for serializing fine-tuned Vision-Language Models with all components (vision encoder, projector, language model) and processor.

=== Description ===

Vision Model Saving extends text model saving to include:
* **Vision encoder** weights (if fine-tuned)
* **Projector** weights
* **Language model** weights
* **AutoProcessor** (tokenizer + image processor)

Same save methods apply (lora, merged_16bit) but GGUF export has limited VLM support.

=== Usage ===

Save VLMs after training. Note that GGUF export for VLMs is limited to certain architectures.

== Theoretical Basis ==

=== VLM Components to Save ===

| Component | LoRA Save | Merged Save |
|-----------|-----------|-------------|
| Vision Encoder | If adapted | Full weights |
| Projector | Full | Full |
| Language Model | LoRA weights | Merged weights |
| Processor | Config only | Config only |

=== GGUF Limitations ===

llama.cpp has limited VLM support. Currently supported:
* LLaVA-style architectures
* Qwen2-VL (partial)

Not supported:
* Most newer VLM architectures
* Multi-image models

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Unslothai_Unsloth_save_pretrained_vision]]
