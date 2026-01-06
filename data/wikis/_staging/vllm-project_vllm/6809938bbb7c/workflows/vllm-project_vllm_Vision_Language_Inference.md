{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Doc|Vision Language Models|https://docs.vllm.ai/en/latest/models/vlm.html]]
|-
! Domains
| [[domain::VLM]], [[domain::Multimodal]], [[domain::Vision_Language]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
End-to-end process for running inference on vision-language models (VLMs) that accept both image and text inputs using vLLM.

=== Description ===
This workflow covers multimodal inference using vLLM's vision-language model support. vLLM supports 60+ VLM architectures including LLaVA, Qwen-VL, InternVL, and BLIP-2. The workflow handles image preprocessing, prompt formatting with image placeholders, and text generation conditioned on visual inputs. VLMs can be used for image captioning, visual question answering, OCR, and document understanding.

=== Usage ===
Execute this workflow when you need to process images alongside text prompts for tasks like visual question answering, image description, chart/diagram understanding, or document OCR. This pattern supports single images, multiple images, and video frames as visual inputs.

== Execution Steps ==

=== Step 1: VLM Model Configuration ===
[[step::Principle:vllm-project_vllm_VLM_Model_Configuration]]

Configure the vLLM engine for a vision-language model. Key settings include multimodal limits (`limit_mm_per_prompt`) to control maximum images per request, memory allocation adjustments for image encoder, and model-specific preprocessing options.

'''Configuration parameters:'''
* `model` - VLM model name (e.g., "llava-hf/llava-v1.6-mistral-7b-hf")
* `limit_mm_per_prompt` - Dictionary specifying max images/videos per prompt
* `mm_processor_kwargs` - Model-specific preprocessing options (crop settings, resolution)
* `max_model_len` - Must account for image token expansion
* `trust_remote_code` - Required for some VLM architectures

=== Step 2: Image Input Preparation ===
[[step::Principle:vllm-project_vllm_Image_Input_Preparation]]

Prepare image inputs for the model. Images can be provided as PIL Image objects, URLs, base64-encoded strings, or file paths. vLLM handles image loading, resizing, and preprocessing according to the model's requirements.

'''Input formats:'''
* PIL `Image` objects loaded with Pillow
* HTTP/HTTPS URLs for remote images
* Local file paths for stored images
* Base64-encoded image data
* Video frames for video-understanding models

=== Step 3: Prompt Construction with Image Tokens ===
[[step::Principle:vllm-project_vllm_VLM_Prompt_Construction]]

Construct prompts that include image placeholder tokens. The exact format varies by model architecture - some use `<image>`, others use `<|image_pad|>` or model-specific tokens. Apply the correct chat template for chat-tuned VLMs to ensure proper formatting.

'''Prompt patterns by model type:'''
* LLaVA-style: `<image>\n{question}`
* Qwen-VL: `<|im_start|>user\n<|image_pad|>\n{question}<|im_end|>`
* BLIP-2: `Question: {question} Answer:`
* Chat models: Use tokenizer's `apply_chat_template` with image placeholders

=== Step 4: Multimodal Generation ===
[[step::Principle:vllm-project_vllm_Multimodal_Generation]]

Execute generation with multimodal inputs by passing images alongside text prompts. The engine processes image encodings through the vision encoder, injects visual features into the language model, and generates text conditioned on both modalities.

'''Generation methods:'''
* `llm.generate()` with `multi_modal_data` parameter for offline inference
* `llm.chat()` with image content blocks for chat-formatted requests
* Server API with base64/URL images in message content
* Batch processing supported for multiple image-text pairs

=== Step 5: VLM Output Processing ===
[[step::Principle:vllm-project_vllm_VLM_Output_Processing]]

Process generation results which contain text responses to visual queries. Output format matches standard text generation with `RequestOutput` objects. For tasks like OCR, additional post-processing may be needed to parse structured output.

'''Output considerations:'''
* Standard `RequestOutput` format with generated text
* Some models output special tokens that should be stripped
* OCR models may produce structured output (tables, coordinates)
* Stop tokens may differ from text-only models

== Execution Diagram ==
{{#mermaid:graph TD
    A[VLM Model Configuration] --> B[Image Input Preparation]
    B --> C[Prompt Construction with Image Tokens]
    C --> D[Multimodal Generation]
    D --> E[VLM Output Processing]
}}

== Related Pages ==
* [[step::Principle:vllm-project_vllm_VLM_Model_Configuration]]
* [[step::Principle:vllm-project_vllm_Image_Input_Preparation]]
* [[step::Principle:vllm-project_vllm_VLM_Prompt_Construction]]
* [[step::Principle:vllm-project_vllm_Multimodal_Generation]]
* [[step::Principle:vllm-project_vllm_VLM_Output_Processing]]
