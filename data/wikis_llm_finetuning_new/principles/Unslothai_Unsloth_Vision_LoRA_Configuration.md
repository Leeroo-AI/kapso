# Principle: Vision_LoRA_Configuration

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|LoRA: Low-Rank Adaptation of Large Language Models|https://arxiv.org/abs/2106.09685]]
* [[source::Paper|LLaVA: Large Language and Vision Assistant|https://arxiv.org/abs/2304.08485]]
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
|-
! Domains
| [[domain::Computer_Vision]], [[domain::Parameter_Efficient_Training]], [[domain::LoRA]]
|-
! Last Updated
| [[last_updated::2026-01-12 00:00 GMT]]
|}

== Overview ==

Mechanism for configuring LoRA adapters in vision-language models with fine-grained control over which components to train.

=== Description ===

Vision LoRA Configuration extends standard LoRA injection with controls specific to vision-language models. VLMs have multiple trainable components:

1. **Vision Encoder**: Processes images (e.g., SigLIP, CLIP)
2. **Projection Layer**: Maps vision to language space
3. **Language Model**: Text generation

Users can selectively enable LoRA on:
* Vision layers only (for visual feature learning)
* Language layers only (for text generation)
* Both (for full multimodal adaptation)
* Attention modules, MLP modules, or both

This granularity allows optimizing training for specific tasks while minimizing memory and compute requirements.

=== Usage ===

Use this principle when:
* Fine-tuning VLMs and need control over which components to adapt
* Memory is constrained and you want to train fewer parameters
* The task primarily requires vision or language understanding (not both)
* You want to preserve pre-trained capabilities in certain components

This step follows vision model loading and precedes data preparation.

== Theoretical Basis ==

'''Component-Specific Training:'''
<syntaxhighlight lang="python">
# Pseudo-code for vision-aware LoRA configuration

# Vision encoder layers (e.g., SigLIP transformer blocks)
vision_modules = [
    "visual.attn.q_proj", "visual.attn.k_proj", "visual.attn.v_proj",
    "visual.mlp.fc1", "visual.mlp.fc2",
]

# Language model layers
language_modules = [
    "model.layers.*.self_attn.q_proj", "model.layers.*.self_attn.k_proj",
    "model.layers.*.mlp.gate_proj", "model.layers.*.mlp.up_proj",
]

# Select based on configuration
if finetune_vision_layers:
    target_modules += vision_modules
if finetune_language_layers:
    target_modules += language_modules
</syntaxhighlight>

'''Training Strategy Guidelines:'''
{| class="wikitable"
|-
! Task Type !! Vision Layers !! Language Layers !! Rationale
|-
| OCR/Document || True || True || Both visual and language understanding needed
|-
| Image Captioning || True || True || Visual features drive language generation
|-
| Visual QA || False || True || Visual features usually sufficient, focus on language
|-
| Domain Adaptation || True || False || Adapt vision encoder to new domain
|}

'''Memory Impact:'''
- Language-only: ~50-70% of full LoRA parameters
- Vision-only: ~30-50% of full LoRA parameters
- Both: 100% (full LoRA)

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Unslothai_Unsloth_get_peft_model_vision]]

