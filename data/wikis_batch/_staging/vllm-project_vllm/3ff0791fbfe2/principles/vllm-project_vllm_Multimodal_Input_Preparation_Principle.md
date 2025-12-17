= Multimodal Input Preparation Principle =

{{Metadata
| Knowledge Sources = vLLM multimodal documentation, PIL Image handling, vllm/multimodal/inputs.py
| Domains = Image Processing, Data Preparation, Multimodal AI
| Last Updated = 2025-12-17
}}

== Overview ==

The '''Multimodal Input Preparation Principle''' establishes the pattern for loading, validating, and preparing visual inputs (images, videos, audio) before they can be processed by vision-language models. This principle ensures that multimodal data is in the correct format, accessible, and properly structured for model consumption.

== Description ==

Multimodal input preparation involves converting raw media from various sources into standardized formats that vLLM can process. The principle encompasses:

* '''Multiple Input Formats''': Support for PIL Images, NumPy arrays, file paths, URLs, and base64-encoded data
* '''Format Conversion''': Automatic conversion between supported formats (e.g., PIL to NumPy array)
* '''Validation''': Ensuring inputs meet model requirements (dimensions, color channels, file formats)
* '''Batch Processing''': Handling single items or lists of items for batch inference
* '''Remote Loading''': Support for loading images from URLs with appropriate error handling

The principle emphasizes flexibility in input formats while maintaining type safety and validation to prevent runtime errors during model inference.

=== Core Concepts ===

; ImageItem
: A single image input that can be a PIL Image, NumPy array, torch.Tensor, or MediaWithBytes wrapper. Represents one visual input to the model.

; ModalityData
: Type alias representing either a single data item or a list of items, allowing for flexible batch processing. Can be None if UUID is provided for caching.

; MultiModalDataDict
: Dictionary mapping modality names (e.g., "image", "video", "audio") to their respective data items, supporting multiple modalities in a single request.

== Usage ==

The multimodal input preparation principle is applied when constructing prompts for vision-language models. Users prepare inputs by:

* Loading images using PIL.Image.open() or from URLs
* Creating multimodal data dictionaries with the appropriate modality keys
* Passing prepared inputs as part of prompt dictionaries to LLM.generate()
* Using the assets module for test images and videos

The preparation phase occurs before the prompt is submitted to the model, ensuring all data is valid and accessible.

== Related Pages ==

* [[implemented_by::Implementation:vllm-project_vllm_Image_Loading_API]]
* [[supports::Vision_Language_Multimodal_Inference_Workflow]]
* [[relates_to::Multimodal_Data_Validation_Pattern]]
