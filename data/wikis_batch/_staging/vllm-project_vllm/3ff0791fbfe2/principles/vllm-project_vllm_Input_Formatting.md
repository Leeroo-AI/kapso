# Input Formatting

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Doc|vLLM Documentation|https://docs.vllm.ai]]
|-
! Domains
| [[domain::Inference]], [[domain::Input_Processing]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:30 GMT]]
|}

== Overview ==

Principle for preparing and structuring prompts in the correct format for LLM inference, supporting text, tokens, and multimodal inputs.

=== Description ===

Input Formatting defines how user inputs are transformed into the internal representations required by the inference engine. vLLM supports multiple input formats:

1. **Raw strings**: Plain text automatically tokenized
2. **TextPrompt**: Dictionary with prompt text and optional multimodal data
3. **TokensPrompt**: Pre-tokenized inputs for maximum control
4. **EmbedsPrompt**: Direct embedding inputs (advanced usage)

Proper input formatting ensures:
* Correct tokenization with model-specific handling
* Efficient batching of heterogeneous inputs
* Support for multimodal data (images, audio) when applicable
* Cache-friendly prompt representation for prefix caching

=== Usage ===

Format inputs when:
* Preparing prompts for batch inference
* Including multimodal data (images, audio)
* Using pre-tokenized inputs for performance
* Implementing custom prompt pipelines

For simple text-only inference, raw strings are sufficient. Use structured formats for advanced scenarios.

== Theoretical Basis ==

'''Input Type Hierarchy:'''
<syntaxhighlight lang="python">
# Type hierarchy for prompts
SingletonPrompt = str | TextPrompt | TokensPrompt | EmbedsPrompt

PromptType = (
    SingletonPrompt                    # Decoder-only models
    | ExplicitEncoderDecoderPrompt     # Encoder-decoder models
    | DataPrompt                       # Plugin-handled formats
)
</syntaxhighlight>

'''Input Resolution:'''
1. String inputs → Converted to TextPrompt internally
2. TextPrompt → Tokenized using model tokenizer
3. TokensPrompt → Used directly (bypass tokenization)
4. ExplicitEncoderDecoderPrompt → Separate encoder/decoder handling

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:vllm-project_vllm_PromptType]]
