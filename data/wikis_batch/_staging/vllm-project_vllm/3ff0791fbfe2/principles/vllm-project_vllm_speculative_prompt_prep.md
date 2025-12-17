= Speculative Prompt Preparation Principle =

== Metadata ==
{| class="wikitable"
|-
! Field !! Value
|-
| Knowledge Sources || vLLM Input Processing Documentation, TokensPrompt API, Speculative Decoding Design
|-
| Domains || Input Processing, Token Management, Speculative Decoding
|-
| Last Updated || 2025-12-17
|}

== Overview ==

The Speculative Prompt Preparation principle defines how input prompts are processed and formatted for speculative decoding workflows. This includes tokenization, prompt formatting, and preparation of token sequences that enable efficient speculation across various methods (ngram, EAGLE, etc.).

== Description ==

Effective speculative decoding depends on properly prepared input sequences that enable pattern matching (for ngram), feature extraction (for EAGLE), or context analysis (for suffix decoding). This principle addresses the unique requirements of speculative methods while maintaining compatibility with standard vLLM input processing.

=== Core Concepts ===

* '''Token-Based Input''': Speculative methods often work directly with token IDs for efficiency
* '''Prompt History''': Some methods (ngram, suffix) benefit from maintaining full prompt history
* '''Multi-Modal Support''': Speculative decoding must handle text, images, and other modalities
* '''Batch Consistency''': All prompts in a batch must be processed uniformly for speculation

=== Input Format Options ===

==== Text Prompts ====
Standard string inputs that are tokenized internally:
<syntaxhighlight lang="python">
prompts = ["The capital of France is", "Write a Python function"]
outputs = llm.generate(prompts, sampling_params)
</syntaxhighlight>

==== TokensPrompt ====
Pre-tokenized input for direct token manipulation:
<syntaxhighlight lang="python">
from vllm.inputs import TokensPrompt
prompts = [TokensPrompt(prompt_token_ids=[1, 2, 3, 4])]
outputs = llm.generate(prompts, sampling_params)
</syntaxhighlight>

==== Chat Format ====
Structured conversations with proper template application:
<syntaxhighlight lang="python">
messages = [{"role": "user", "content": "Hello"}]
outputs = llm.chat([messages], sampling_params)
</syntaxhighlight>

=== Method-Specific Considerations ===

==== N-gram Speculation ====
* Requires access to full prompt token sequence
* Benefits from longer prompts with repetitive patterns
* No special preprocessing needed
* Token history maintained for pattern matching

==== EAGLE Speculation ====
* Needs hidden states from target model's forward pass
* Prompt length affects tree attention metadata
* May require padding in batched scenarios
* Multi-modal inputs supported through embeddings

==== MLP Speculator ====
* Works with context vectors and sampled tokens
* No special prompt requirements
* Efficient with standard tokenization

==== Suffix Decoding ====
* Benefits from maintaining generation history across requests
* Builds global and per-prompt suffix trees
* Requires consistent tokenization across related requests

=== Tokenization Requirements ===

* '''Consistency''': Target and draft models must use same tokenizer
* '''Special Tokens''': Proper handling of BOS, EOS, and padding tokens
* '''Vocabulary Alignment''': Token IDs must be compatible across models
* '''Detokenization''': Round-trip consistency for text → tokens → text

== Usage Context ==

This principle applies when:

* Preparing inputs for speculative generate() calls
* Optimizing prompts for specific speculative methods
* Handling multi-modal inputs with speculation
* Processing large batches with mixed prompt lengths

Proper prompt preparation ensures:
* Maximum speculation effectiveness
* Correct token sequence handling
* Efficient memory usage
* Compatibility across all speculative methods

== Design Considerations ==

=== Trade-offs ===

* '''Pre-tokenization vs. Auto-tokenization''': TokensPrompt gives control but requires manual tokenization
* '''Prompt Length vs. Memory''': Longer prompts enable better ngram matching but use more memory
* '''Batch Size vs. Padding''': Larger batches may require padding, affecting EAGLE efficiency
* '''Format vs. Flexibility''': Structured formats (chat) may limit speculation opportunities

=== Best Practices ===

* '''Use TokensPrompt''': When you need precise control over token sequences
* '''Leverage Chat Templates''': For conversational applications with speculation
* '''Batch Similar Lengths''': Minimize padding overhead in EAGLE batches
* '''Preserve History''': For suffix decoding, maintain request context

=== Common Pitfalls ===

* '''Tokenizer Mismatch''': Using different tokenizers for target and draft models
* '''Truncation''': Cutting prompts too short reduces ngram effectiveness
* '''Padding Inconsistency''': Improper padding breaks EAGLE tree attention
* '''Special Token Errors''': Incorrect handling of BOS/EOS affects generation quality

== Integration with Speculative Methods ==

=== N-gram Method ===
* Prompt tokens stored in circular buffer
* Pattern matching against last N tokens
* No special formatting required
* Works with any prompt format

=== EAGLE Method ===
* Prompt processed through target model for hidden states
* Tree attention requires consistent padding (or disable_padded_drafter_batch)
* Multi-modal prompts converted to embeddings
* Batch preparation coordinates across sequences

=== MLP Speculator ===
* Standard prompt processing
* Context vectors extracted from target model
* No special requirements

=== Suffix Method ===
* Builds per-prompt suffix tree from tokens
* Maintains global cache across requests
* Benefits from consistent prompt structure

== Related Principles ==

* [[implements::vllm-project_vllm_TokensPrompt_spec]] - TokensPrompt implementation
* Tokenization Principles - Token processing fundamentals
* Batch Processing Principles - Efficient batching strategies
* Multi-Modal Input Principles - Handling diverse input types

== See Also ==

* [[implemented_by::Implementation:vllm-project_vllm_TokensPrompt_spec]]
* [[implements::vllm-project_vllm_LLM_generate_spec]]
* vLLM Input Types Documentation
* Chat Template Documentation
