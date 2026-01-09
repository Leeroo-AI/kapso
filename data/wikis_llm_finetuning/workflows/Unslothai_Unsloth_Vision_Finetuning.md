# Workflow: Vision_Finetuning

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|Vision Fine-tuning|https://unsloth.ai/docs/basics/vision-fine-tuning]]
* [[source::Doc|VLM Guide|https://unsloth.ai/blog/vision]]
|-
! Domains
| [[domain::LLMs]], [[domain::Vision]], [[domain::Multimodal]], [[domain::Fine_Tuning]]
|-
! Last Updated
| [[last_updated::2026-01-09 16:00 GMT]]
|}

== Overview ==
End-to-end process for fine-tuning Vision-Language Models (VLMs) using Unsloth's optimized pipeline for multimodal understanding tasks like OCR, image captioning, and visual question answering.

=== Description ===
This workflow enables fine-tuning of Vision-Language Models (VLMs) such as Qwen2-VL, Llama 3.2 Vision, and Pixtral on custom image-text datasets. Unsloth supports training both the vision encoder and language model components with memory-efficient techniques.

The workflow covers:
1. **Vision Model Loading**: Loading VLMs with quantization and vision preprocessing
2. **Vision LoRA Configuration**: Configuring adapters for both vision and language layers
3. **Multimodal Data Preparation**: Formatting image-text pairs with proper message structure
4. **Vision Training**: Fine-tuning using UnslothVisionDataCollator for proper batching
5. **Model Export**: Saving merged multimodal models for deployment

Supported Vision Models:
* Qwen2-VL (2B, 7B, 72B)
* Qwen2.5-VL (all sizes)
* Llama 3.2 Vision (11B, 90B)
* Pixtral (12B)
* DeepSeek-OCR

=== Usage ===
Execute this workflow when:
* You need to fine-tune a VLM on custom image-text data
* Your task involves OCR, document understanding, or visual QA
* You want to improve image captioning or visual reasoning capabilities
* You need a specialized multimodal model for domain-specific images

'''Input requirements:'''
* Vision-Language Model (Qwen2-VL, Llama 3.2 Vision, etc.)
* Training dataset with images and text conversations
* CUDA-capable GPU with 16GB+ VRAM (for 7B models)

'''Expected outputs:'''
* Fine-tuned VLM with improved task-specific performance
* LoRA adapter or merged model for deployment
* OCR accuracy improvements on domain-specific documents

== Execution Steps ==

=== Step 1: Vision Model Loading ===
[[step::Principle:Unslothai_Unsloth_Vision_Model_Loading]]

Initialize the Vision-Language Model using `FastVisionModel.from_pretrained()`. This loads both the vision encoder and language model with appropriate quantization settings.

'''Key considerations:'''
* Use `FastVisionModel` instead of `FastLanguageModel` for VLMs
* Set `load_in_4bit=True` for memory-efficient training
* Choose appropriate `max_seq_length` for your image-text data
* Vision models require more VRAM than text-only models

=== Step 2: Vision LoRA Configuration ===
[[step::Principle:Unslothai_Unsloth_Vision_LoRA_Configuration]]

Configure LoRA adapters for vision model fine-tuning using `FastVisionModel.get_peft_model()`. This allows selective training of vision encoder and/or language model components.

'''Key considerations:'''
* Set `finetune_vision_layers=True` to train vision encoder
* Set `finetune_language_layers=True` to train LLM components
* Use `finetune_attention_modules=True` for attention layer adaptation
* Use `finetune_mlp_modules=True` for MLP layer adaptation
* Enable `use_gradient_checkpointing="unsloth"` for memory efficiency

=== Step 3: Multimodal Dataset Preparation ===
[[step::Principle:Unslothai_Unsloth_Multimodal_Data_Preparation]]

Format the training dataset with proper multimodal message structure. Each example should include images and text in the OpenAI-compatible message format.

'''Message structure:'''
```
{
    "messages": [
        {"role": "system", "content": [{"type": "text", "text": "..."}]},
        {"role": "user", "content": [
            {"type": "text", "text": "..."},
            {"type": "image", "image": <PIL.Image>}
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": "..."}]}
    ]
}
```

'''Key considerations:'''
* Use PIL.Image objects for images (not file paths)
* Structure content as list of type/text or type/image dicts
* Include system prompts to guide model behavior
* Process images using model-specific utilities (e.g., qwen_vl_utils)

=== Step 4: Vision Training Mode ===
[[step::Principle:Unslothai_Unsloth_Vision_Training_Mode]]

Set the model to training mode and configure the trainer with vision-specific settings using `FastVisionModel.for_training()`.

'''Key considerations:'''
* Call `FastVisionModel.for_training(model)` before training
* Disable `use_cache` for training compatibility
* Use `UnslothVisionDataCollator` for proper batching
* Set `remove_unused_columns=False` in training config

=== Step 5: Vision Model Training ===
[[step::Principle:Unslothai_Unsloth_Vision_Training]]

Execute the training loop using SFTTrainer with vision-specific configuration. The UnslothVisionDataCollator handles image preprocessing and batching.

'''Key considerations:'''
* Use smaller batch sizes than text-only training (2-4)
* Enable gradient checkpointing with `use_reentrant=False`
* Set `dataset_kwargs={"skip_prepare_dataset": True}`
* Monitor OCR/visual QA metrics during training
* Use `dataset_text_field=""` (empty string for vision)

=== Step 6: Vision Model Inference Mode ===
[[step::Principle:Unslothai_Unsloth_Vision_Inference_Mode]]

Switch to inference mode for evaluation using `FastVisionModel.for_inference()`. This optimizes the model for generation tasks.

'''Key considerations:'''
* Call `FastVisionModel.for_inference(model)` before evaluation
* Enable `use_cache` for efficient generation
* Use `process_vision_info()` for proper image preprocessing during inference

=== Step 7: Model Saving and Evaluation ===
[[step::Principle:Unslothai_Unsloth_Vision_Model_Saving]]

Save the trained vision model and evaluate on relevant benchmarks. Vision models support the same export options as text models.

'''Key considerations:'''
* Save LoRA adapter with `save_pretrained()`
* Merge with `save_pretrained_merged()` for deployment
* Evaluate on OCR benchmarks (WER, CER metrics)
* Test on domain-specific visual QA tasks

== Execution Diagram ==
{{#mermaid:graph TD
    A[Vision Model Loading] --> B[Vision LoRA Configuration]
    B --> C[Multimodal Dataset Preparation]
    C --> D[Vision Training Mode]
    D --> E[Vision Model Training]
    E --> F[Vision Inference Mode]
    F --> G[Evaluation]
    G --> H[Model Saving]
    H --> I{Export Format}
    I -->|LoRA| J[Vision Adapter]
    I -->|Merged| K[Full VLM]
}}

== Related Pages ==
* [[step::Principle:Unslothai_Unsloth_Vision_Model_Loading]]
* [[step::Principle:Unslothai_Unsloth_Vision_LoRA_Configuration]]
* [[step::Principle:Unslothai_Unsloth_Multimodal_Data_Preparation]]
* [[step::Principle:Unslothai_Unsloth_Vision_Training_Mode]]
* [[step::Principle:Unslothai_Unsloth_Vision_Training]]
* [[step::Principle:Unslothai_Unsloth_Vision_Inference_Mode]]
* [[step::Principle:Unslothai_Unsloth_Vision_Model_Saving]]
