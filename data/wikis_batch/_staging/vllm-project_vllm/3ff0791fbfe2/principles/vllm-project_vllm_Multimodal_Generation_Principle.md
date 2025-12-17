= Multimodal Generation Principle =

{{Metadata
| Knowledge Sources = vllm/entrypoints/llm.py generate() method, generation pipeline analysis
| Domains = Text Generation, Inference, Multimodal Processing
| Last Updated = 2025-12-17
}}

== Overview ==

The '''Multimodal Generation Principle''' defines the pattern for executing inference on vision-language models, processing both visual and textual inputs through the model to produce text outputs. This principle ensures that multimodal data is properly integrated into the generation pipeline and that outputs are produced efficiently.

== Description ==

Multimodal generation extends standard text generation by incorporating visual features into the model's input representation. The principle encompasses:

* '''Input Processing''': Converting text and images into a unified sequence of embeddings
* '''Feature Integration''': Merging visual features from the encoder with text token embeddings
* '''Autoregressive Generation''': Producing output tokens conditioned on both visual and textual context
* '''Batch Processing''': Handling multiple multimodal requests efficiently with batching
* '''Sampling Control''': Applying temperature, top-p, top-k, and other sampling strategies to outputs
* '''Progress Tracking''': Monitoring generation progress through optional progress bars

The principle emphasizes that multimodal generation follows the same autoregressive pattern as text-only generation, but with additional preprocessing to handle visual inputs.

=== Core Concepts ===

; Multimodal Prompt
: A dictionary containing both a text prompt string and multimodal data (images, videos), structured for model processing.

; Sampling Parameters
: Configuration controlling output generation behavior including temperature, max tokens, stop sequences, and sampling strategies.

; Request Output
: The generated text along with metadata such as token IDs, finish reason, and usage statistics.

== Usage ==

The multimodal generation principle is applied when calling `llm.generate()` with multimodal prompts. Users:

* Prepare prompts with both text and multimodal data
* Configure sampling parameters for desired output characteristics
* Submit prompts to the model via `generate()`
* Receive `RequestOutput` objects containing generated text
* Can process multiple prompts in batch for efficiency

Generation occurs after initialization and can be called repeatedly with different inputs.

== Related Pages ==

* [[implemented_by::Implementation:vllm-project_vllm_LLM_Generate_Multimodal_API]]
* [[supports::Vision_Language_Multimodal_Inference_Workflow]]
* [[relates_to::Autoregressive_Generation_Pattern]]
