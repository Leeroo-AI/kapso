= VLM Configuration Principle =

{{Metadata
| Knowledge Sources = vLLM codebase analysis, EngineArgs documentation, VLM examples
| Domains = Vision-Language Models, Model Configuration, Multimodal AI
| Last Updated = 2025-12-17
}}

== Overview ==

The '''VLM Configuration Principle''' defines the architectural pattern for configuring vision-language models in vLLM. This principle establishes that multimodal model initialization requires explicit specification of model parameters, multimodal processor settings, and resource allocation constraints to handle both visual and textual inputs effectively.

== Description ==

Vision-language model configuration in vLLM follows a principled approach that separates model loading parameters from runtime behavior. The configuration pattern includes:

* '''Model Identification''': Specifying the HuggingFace model path or name
* '''Multimodal Processor Configuration''': Setting parameters for image/video processing via `mm_processor_kwargs`
* '''Resource Limits''': Defining constraints on multimodal inputs per prompt using `limit_mm_per_prompt`
* '''Memory Management''': Configuring GPU memory utilization and KV cache settings
* '''Trust and Security''': Enabling remote code execution when required by model architectures

This principle ensures that vision-language models are properly initialized with the necessary computational resources and processing capabilities before inference begins.

=== Core Concepts ===

; Multimodal Model Parameters
: Configuration parameters that control how visual and textual modalities are processed together, including processor-specific overrides and memory allocation.

; Limit Specification
: Constraints on the number of images, videos, or other multimodal inputs that can be included in a single prompt, preventing resource exhaustion.

; Processor Kwargs
: Model-specific overrides passed to the HuggingFace processor, controlling aspects like image cropping, resolution, number of frames, etc.

== Usage ==

The VLM configuration principle is applied during model initialization, before any inference operations. Configuration parameters are passed through the `EngineArgs` class or directly to the `LLM` constructor. Common configurations include:

* Setting `trust_remote_code=True` for models with custom architectures
* Configuring `mm_processor_kwargs` for vision-specific processing (e.g., `{"num_crops": 16}` for Phi-3-Vision)
* Defining `limit_mm_per_prompt` to control resource usage (e.g., `{"image": 1, "video": 0}`)
* Adjusting `max_model_len` and `gpu_memory_utilization` for model size constraints

== Related Pages ==

* [[implemented_by::Implementation:vllm-project_vllm_EngineArgs_Multimodal_API]]
* [[supports::Vision_Language_Multimodal_Inference_Workflow]]
* [[relates_to::Multimodal_Input_Processing_Pattern]]
