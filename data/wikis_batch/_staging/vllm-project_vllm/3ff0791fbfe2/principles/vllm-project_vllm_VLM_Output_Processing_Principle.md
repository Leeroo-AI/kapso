= VLM Output Processing Principle =

{{Metadata
| Knowledge Sources = vllm/outputs.py, output handling patterns
| Domains = Output Handling, Result Processing, Data Structures
| Last Updated = 2025-12-17
}}

== Overview ==

The '''VLM Output Processing Principle''' establishes the pattern for handling and extracting information from vision-language model outputs. This principle ensures that generated text, metadata, and performance metrics are properly structured and accessible to users.

== Description ==

VLM output processing involves packaging generation results into structured objects that provide both the generated text and useful metadata. The principle encompasses:

* '''Output Structure''': Organizing results into standardized `RequestOutput` and `CompletionOutput` objects
* '''Text Extraction''': Providing easy access to generated text strings
* '''Token Information''': Exposing token IDs and logprobs for analysis
* '''Metadata Access''': Including finish reasons, stop conditions, and usage statistics
* '''Batch Handling''': Maintaining correspondence between inputs and outputs in batch processing
* '''Multiple Completions''': Supporting n-best outputs when using beam search or multiple samples

The principle emphasizes that outputs should be self-contained, with all information needed to understand and use the results.

=== Core Concepts ===

; RequestOutput
: Container for all outputs from a single request, including prompt information, generated completions, and metadata.

; CompletionOutput
: Individual completion within a request, containing the generated text, tokens, and associated scores.

; Finish Reason
: Indication of why generation stopped (e.g., "stop" for stop sequence, "length" for max tokens, "eos_token" for end-of-sequence).

== Usage ==

The VLM output processing principle is applied after generation completes. Users:

* Receive `List[RequestOutput]` from `generate()` calls
* Access generated text via `output.outputs[0].text`
* Inspect token IDs via `output.outputs[0].token_ids`
* Check completion status via `output.finished`
* Extract metadata like finish reasons and logprobs
* Process multiple completions if n > 1 or beam search is used

Output processing occurs synchronously after generation, with all results immediately available.

== Related Pages ==

* [[implemented_by::Implementation:vllm-project_vllm_RequestOutput_VLM_API]]
* [[supports::Vision_Language_Multimodal_Inference_Workflow]]
* [[relates_to::Generation_Result_Structures]]
