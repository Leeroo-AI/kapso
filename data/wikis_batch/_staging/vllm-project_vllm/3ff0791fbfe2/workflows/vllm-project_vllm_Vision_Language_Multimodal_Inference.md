{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Doc|vLLM Documentation|https://docs.vllm.ai]]
|-
! Domains
| [[domain::LLMs]], [[domain::Multimodal]], [[domain::Vision]], [[domain::Inference]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==

End-to-end process for running inference on vision-language models (VLMs) that process both text and image inputs for tasks like image captioning, visual question answering, and document understanding.

=== Description ===

This workflow demonstrates multimodal inference with vLLM, supporting 60+ vision-language models including LLaVA, Qwen-VL, InternVL, and many others. The process covers loading multimodal models, formatting prompts with image placeholders, and generating text responses conditioned on visual inputs.

Key capabilities:
* **60+ VLM architectures**: LLaVA, Qwen-VL, InternVL, Phi-3-Vision, etc.
* **Multiple image inputs**: Support for multi-image conversations
* **Video support**: Process video frames for video understanding
* **Image preprocessing**: Automatic resizing, cropping, and normalization
* **Chat templates**: Model-specific prompt formatting with image tokens

=== Usage ===

Execute this workflow when you need to:
* Generate captions or descriptions for images
* Answer questions about visual content
* Extract information from documents and diagrams
* Analyze multiple images in a single conversation
* Process video content frame by frame

Ideal for applications requiring visual understanding alongside language generation.

== Execution Steps ==

=== Step 1: Select and Configure VLM ===
[[step::Principle:vllm-project_vllm_VLM_Configuration_Principle]]

Choose a vision-language model and configure engine arguments with multimodal-specific settings. Different VLM architectures have different requirements for image limits, context length, and processor configurations.

'''Key considerations:'''
* Set `limit_mm_per_prompt` to specify maximum images/videos per request
* Configure `mm_processor_kwargs` for model-specific preprocessing
* Set appropriate `max_model_len` for multimodal context
* Enable `trust_remote_code` for custom model implementations

=== Step 2: Prepare Multimodal Inputs ===
[[step::Principle:vllm-project_vllm_Multimodal_Input_Preparation_Principle]]

Load and prepare image or video inputs. Images can be provided as PIL Image objects, URLs, base64-encoded strings, or file paths. The processor handles resizing, normalization, and conversion to model-expected formats.

'''Input sources:'''
* Local file paths
* HTTP/HTTPS URLs
* Base64-encoded image data
* PIL Image objects
* Video files (extracted as frames)

=== Step 3: Format Multimodal Prompts ===
[[step::Principle:vllm-project_vllm_Multimodal_Prompt_Formatting_Principle]]

Construct prompts with model-specific image placeholder tokens. Each VLM architecture uses different tokens (e.g., `<image>`, `<|image|>`, `<fim_prefix><|img|><fim_suffix>`) and prompt structures.

'''Prompt patterns by model:'''
* LLaVA: `<image>\n{question}`
* Qwen-VL: `<|im_start|>user\n<image>\n{question}<|im_end|>`
* InternVL: `<image>\n{question}`
* Model-specific templates available in examples

=== Step 4: Initialize VLM Engine ===
[[step::Principle:vllm-project_vllm_VLM_Engine_Initialization_Principle]]

Create the LLM instance with multimodal configuration. The engine loads both the language model and vision encoder, initializes the image processor, and prepares the multimodal embedding layer.

'''Initialization process:'''
1. Load vision encoder weights (e.g., CLIP, SigLIP)
2. Load language model weights
3. Initialize multimodal projection layers
4. Set up image preprocessing pipeline
5. Allocate KV cache with multimodal token budget

=== Step 5: Execute Multimodal Generation ===
[[step::Principle:vllm-project_vllm_Multimodal_Generation_Principle]]

Submit prompts with attached multimodal data for generation. The engine processes images through the vision encoder, projects embeddings into language space, and generates text conditioned on both modalities.

'''Generation flow:'''
1. Images preprocessed and encoded by vision model
2. Image embeddings projected to language model dimension
3. Text tokens and image embeddings combined
4. Autoregressive generation produces output text

=== Step 6: Process VLM Outputs ===
[[step::Principle:vllm-project_vllm_VLM_Output_Processing_Principle]]

Extract generated text from the multimodal inference results. Outputs follow the same structure as text-only generation but may include model-specific formatting or special tokens.

'''Output handling:'''
* Strip model-specific end tokens if needed
* Handle multi-turn context for conversations
* Extract structured information from descriptions

== Execution Diagram ==
{{#mermaid:graph TD
    A[Select and Configure VLM] --> B[Prepare Multimodal Inputs]
    B --> C[Format Multimodal Prompts]
    C --> D[Initialize VLM Engine]
    D --> E[Execute Multimodal Generation]
    E --> F[Process VLM Outputs]
}}

== Related Pages ==
* [[step::Principle:vllm-project_vllm_VLM_Configuration_Principle]]
* [[step::Principle:vllm-project_vllm_Multimodal_Input_Preparation_Principle]]
* [[step::Principle:vllm-project_vllm_Multimodal_Prompt_Formatting_Principle]]
* [[step::Principle:vllm-project_vllm_VLM_Engine_Initialization_Principle]]
* [[step::Principle:vllm-project_vllm_Multimodal_Generation_Principle]]
* [[step::Principle:vllm-project_vllm_VLM_Output_Processing_Principle]]
