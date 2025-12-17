= Multimodal Prompt Formatting Principle =

{{Metadata
| Knowledge Sources = VLM examples, model documentation, vllm/multimodal/processing.py
| Domains = Prompt Engineering, Vision-Language Models, Template Design
| Last Updated = 2025-12-17
}}

== Overview ==

The '''Multimodal Prompt Formatting Principle''' defines the pattern for structuring prompts that combine textual instructions with visual placeholders for vision-language models. This principle ensures that models correctly interpret where visual features should be integrated into the text sequence.

== Description ==

Multimodal prompt formatting requires careful attention to model-specific template conventions. The principle encompasses:

* '''Placeholder Tokens''': Special tokens that indicate where image/video features should be inserted (e.g., `<image>`, `<|image_pad|>`, `<IMG>`)
* '''Chat Templates''': Structured conversation formats with role markers (e.g., `USER:`, `<|im_start|>user`)
* '''Position Sensitivity''': Correct placement of placeholders relative to instructions affects model understanding
* '''Model Variations''': Different VLM architectures use different placeholder conventions
* '''Multi-Image Support''': Some models support multiple placeholders for comparing or reasoning across images

The principle emphasizes that incorrect prompt formatting can lead to poor model performance, as the model may not correctly locate visual features in the input sequence.

=== Core Concepts ===

; Image Placeholder
: A special token or sequence that marks where visual features should be embedded in the text sequence. Examples include `<image>`, `<|image_pad|>`, `<IMG>`, `[IMG]`.

; Chat Template
: A structured format for conversation that includes system messages, user queries, and assistant responses. VLMs typically use these templates to maintain conversation context.

; Placeholder Position
: The location of image placeholders relative to textual instructions. Common patterns include prefix (image before text), suffix (image after text), or interleaved (images mixed with text).

== Usage ==

The multimodal prompt formatting principle is applied when constructing the `prompt` string in the input dictionary. Users must:

* Consult the model's documentation or HuggingFace card for the correct template
* Include appropriate image placeholders in the correct positions
* Follow the model's chat template format if applicable
* Ensure placeholder count matches the number of images provided
* Use consistent formatting across batch requests

The formatting occurs after images are loaded but before the prompt is submitted to the model.

== Related Pages ==

* [[implemented_by::Implementation:vllm-project_vllm_VLM_Prompt_Templates_Pattern]]
* [[supports::Vision_Language_Multimodal_Inference_Workflow]]
* [[relates_to::Chat_Template_Processing_Pattern]]
