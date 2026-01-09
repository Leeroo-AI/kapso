# Principle: Vision_Model_Loading

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|LLaVA|https://arxiv.org/abs/2304.08485]]
* [[source::Paper|Qwen-VL|https://arxiv.org/abs/2308.12966]]
|-
! Domains
| [[domain::Computer_Vision]], [[domain::NLP]], [[domain::Multimodal]]
|-
! Last Updated
| [[last_updated::2026-01-09 16:00 GMT]]
|}

== Overview ==

Technique for loading Vision-Language Models with memory-efficient quantization while preserving multimodal understanding capabilities.

=== Description ===

Vision Model Loading prepares VLMs for multimodal fine-tuning by:

1. Loading vision encoder (often ViT-based)
2. Loading language model (transformer decoder)
3. Loading projector/connector between modalities
4. Quantizing applicable components

VLMs have unique considerations:
* Vision encoders often stay in higher precision
* Language model benefits most from quantization
* Projector layers may be trained from scratch

=== Usage ===

Load VLMs when fine-tuning for:
* Image-text tasks (captioning, VQA)
* Document understanding (OCR, charts)
* Multi-image reasoning
* Video understanding (frame-based)

Key differences from text models:
* Returns AutoProcessor (not tokenizer)
* Handles image preprocessing
* May have separate vision/language LoRA options

== Theoretical Basis ==

=== VLM Architecture ===

Typical VLM structure:

<math>
\text{Image} \xrightarrow{\text{Vision Encoder}} \text{Visual Tokens} \xrightarrow{\text{Projector}} \text{Language Space} \xrightarrow{\text{LLM}} \text{Output}
</math>

Components:
* **Vision Encoder**: Extracts image features (ViT, SigLIP)
* **Projector**: Maps visual to language space (MLP, cross-attention)
* **LLM**: Generates text conditioned on visual + text tokens

=== Quantization Strategy ===

Different components have different quantization sensitivity:

| Component | Quantization | Reason |
|-----------|--------------|--------|
| Vision Encoder | Keep higher precision | Visual features sensitive |
| Projector | Trainable (not quantized) | Small, critical |
| LLM | 4-bit NF4 | Large, benefits most |

=== Memory Estimation ===

<math>
\text{Memory} = \text{Vision}_{fp16} + \text{Projector}_{fp16} + \text{LLM}_{4bit}
</math>

For 11B VLM:
* Vision: ~600MB (ViT-L/14)
* Projector: ~200MB
* LLM: ~6GB (4-bit)
* Total: ~7GB (vs 22GB at fp16)

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Unslothai_Unsloth_FastVisionModel_from_pretrained]]
