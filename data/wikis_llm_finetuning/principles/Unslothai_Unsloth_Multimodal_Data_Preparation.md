# Principle: Multimodal_Data_Preparation

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|LLaVA|https://arxiv.org/abs/2304.08485]]
* [[source::Doc|HuggingFace Datasets|https://huggingface.co/docs/datasets]]
|-
! Domains
| [[domain::Computer_Vision]], [[domain::Data_Engineering]], [[domain::Multimodal]]
|-
! Last Updated
| [[last_updated::2026-01-09 16:00 GMT]]
|}

== Overview ==

Technique for formatting image-text datasets into the structure expected by Vision-Language Model training pipelines.

=== Description ===

Multimodal Data Preparation creates datasets with interleaved image and text content. Unlike text-only datasets, VLM datasets must:

1. **Include images** as PIL.Image objects or paths
2. **Mark image positions** in the message content
3. **Handle multiple images** per conversation
4. **Support multi-turn** conversations with images at any turn

=== Usage ===

Prepare multimodal data when:
* Fine-tuning VLMs on custom image-text pairs
* Training for VQA, captioning, or document tasks
* Creating instruction-following datasets with visual context

== Theoretical Basis ==

=== Message Structure ===

VLM messages use content arrays with typed elements:

<syntaxhighlight lang="python">
{
    "role": "user",
    "content": [
        {"type": "image"},           # Image placeholder
        {"type": "text", "text": "..."} # Text content
    ]
}
</syntaxhighlight>

The processor converts this to:
* Image → Visual tokens (e.g., 576 tokens for ViT-L/14)
* Text → Text tokens

=== Image Processing ===

AutoProcessor handles:
* Resizing to model's expected resolution
* Normalization (mean, std)
* Conversion to tensor format
* Padding/batching for variable-size images

=== Data Flow ===

<math>
\text{Raw Data} \xrightarrow{\text{Formatting}} \text{Messages + Images} \xrightarrow{\text{Processor}} \text{Tokens + Pixel Values}
</math>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Unslothai_Unsloth_multimodal_dataset_pattern]]
