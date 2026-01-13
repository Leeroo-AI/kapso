# Workflow: Vision_Model_Finetuning

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|Qwen2-VL Documentation|https://qwenlm.github.io/blog/qwen2-vl/]]
|-
! Domains
| [[domain::Vision_Language_Models]], [[domain::Fine_Tuning]], [[domain::Multimodal]]
|-
! Last Updated
| [[last_updated::2026-01-12 19:00 GMT]]
|}

== Overview ==

End-to-end process for fine-tuning Vision-Language Models (VLMs) on image-text tasks using Unsloth's optimized training pipeline with selective layer freezing.

=== Description ===

This workflow enables fine-tuning of multimodal Vision-Language Models such as Qwen2-VL, Llama 3.2 Vision, and Pixtral. Unsloth's FastVisionModel provides the same 2x speedup and memory optimizations available for language-only models.

The workflow supports:
* Selective fine-tuning (vision encoder, language layers, or both)
* Image-text training with proper multimodal data collation
* OCR, visual question answering, and document understanding tasks
* Model merging and deployment

=== Usage ===

Execute this workflow when you have an image-text dataset and need to adapt a vision-language model for specific tasks such as:
* Optical Character Recognition (OCR)
* Visual Question Answering (VQA)
* Document understanding and extraction
* Image captioning with domain-specific terminology

== Execution Steps ==

=== Step 1: Vision_Model_Loading ===

Load a vision-language model using FastVisionModel with appropriate quantization settings. The loader handles both the vision encoder and language model components.

'''Key considerations:'''
* Use `FastVisionModel.from_pretrained()` for VLMs
* Set `load_in_4bit=True` for memory efficiency
* Configure `max_seq_length` considering both image tokens and text
* Supported models: Qwen2-VL, Llama 3.2 Vision, Pixtral

=== Step 2: Vision_LoRA_Configuration ===

Configure LoRA adapters with fine-grained control over which components to fine-tune. Vision models allow separate control of vision encoder, language layers, attention, and MLP modules.

'''Configuration options:'''
* `finetune_vision_layers` - Train vision encoder components
* `finetune_language_layers` - Train language model components
* `finetune_attention_modules` - Target attention projections
* `finetune_mlp_modules` - Target feed-forward layers

'''What happens:'''
* LoRA adapters injected into selected modules only
* Unselected components remain frozen during training
* Allows task-specific tuning (e.g., language-only for domain vocabulary)

=== Step 3: Multimodal_Data_Preparation ===

Prepare image-text data in the OpenAI-compatible message format required by vision models. Each sample contains interleaved text and image content.

'''Message structure:'''
* System message with task instructions
* User message with text query and image reference
* Assistant message with expected response

'''What happens:'''
* Images processed through vision encoder during training
* Text and image tokens interleaved in model input
* Special handling for variable image resolutions

'''Key considerations:'''
* Use PIL Image objects directly (not bytes)
* Match image format to model requirements
* Consider image resolution vs memory tradeoffs

=== Step 4: Vision_Training_Setup ===

Configure training with vision-specific settings. Use UnslothVisionDataCollator for proper multimodal batching.

'''Critical settings:'''
* `UnslothVisionDataCollator` - Required for proper image batching
* `remove_unused_columns=False` - Preserve image data
* `dataset_kwargs={"skip_prepare_dataset": True}` - Custom data handling
* `dataset_text_field=""` - Empty since using message format

'''Training parameters:'''
* Lower batch sizes than text-only due to image memory
* Gradient checkpointing strongly recommended
* Standard LoRA learning rates (2e-4)

=== Step 5: SFT_Vision_Training ===

Execute supervised fine-tuning on the multimodal dataset. The trainer handles image preprocessing and proper gradient flow through selected components.

'''What happens:'''
* Images processed through vision encoder
* Image features projected into language model space
* Loss computed on text tokens only
* Gradients flow through LoRA adapters in selected components

=== Step 6: Vision_Model_Merging ===

Merge trained adapters and export the model. Vision models support the same save methods as language models.

'''Save methods:'''
* `save_pretrained` - Save LoRA adapter only
* `save_pretrained_merged` - Merge into full model

'''Validation:'''
* Test on held-out samples
* Measure task-specific metrics (WER/CER for OCR, accuracy for VQA)
* Compare across loading precisions (4-bit, 8-bit, 16-bit)

== Execution Diagram ==

{{#mermaid:graph TD
    A[Vision_Model_Loading] --> B[Vision_LoRA_Configuration]
    B --> C[Multimodal_Data_Preparation]
    C --> D[Vision_Training_Setup]
    D --> E[SFT_Vision_Training]
    E --> F[Vision_Model_Merging]
}}

== GitHub URL ==

The executable implementation will be available at:

[[github_url::PENDING_REPO_BUILD]]

<!-- This URL will be populated by the repo builder phase -->
