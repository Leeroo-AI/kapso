{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|HuggingFace Tokenizers|https://huggingface.co/docs/transformers/main_classes/tokenizer]]
|-
! Domains
| [[domain::NLP]], [[domain::Tokenization]], [[domain::Text_Processing]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==
End-to-end process for converting raw text into tokenized model inputs and decoding model outputs back to text using the tokenization infrastructure.

=== Description ===
This workflow covers the complete tokenization pipeline in the Transformers library. It handles:

1. **Tokenizer Loading**: Loading slow (Python) or fast (Rust) tokenizers from checkpoints
2. **Text Encoding**: Converting text strings to token IDs with special tokens
3. **Padding and Truncation**: Handling variable-length sequences for batching
4. **Chat Templates**: Applying conversation formatting for chat models
5. **Decoding**: Converting token IDs back to readable text

The library supports both slow Python tokenizers and fast Rust-based tokenizers (via HuggingFace tokenizers library) with multiple algorithms: BPE, WordPiece, Unigram, and SentencePiece.

=== Usage ===
Execute this workflow when you need to:
* Prepare text data for model input
* Decode model outputs to human-readable text
* Apply chat templates for instruction-tuned models
* Handle batch tokenization with padding
* Convert between slow and fast tokenizers

== Execution Steps ==

=== Step 1: Tokenizer Loading ===
[[step::Principle:huggingface_transformers_Tokenizer_Loading]]

Load the tokenizer from a pretrained checkpoint or local files. AutoTokenizer automatically selects the correct tokenizer class based on the model type. Fast tokenizers are preferred when available for performance.

'''Tokenizer types:'''
* Fast tokenizers: Rust-based, 10-100x faster, support offset mappings
* Slow tokenizers: Pure Python, more flexible for customization
* SentencePiece: For models trained with SentencePiece (Llama, T5)
* Custom: Models with trust_remote_code may have custom tokenizers

=== Step 2: Special Tokens Configuration ===
[[step::Principle:huggingface_transformers_Special_Tokens]]

Configure special tokens that control model behavior: BOS (beginning), EOS (end), PAD (padding), UNK (unknown), SEP (separator), CLS (classification), and MASK. These tokens are handled specially during encoding/decoding.

'''Token roles:'''
* BOS/EOS: Mark sequence boundaries for generation
* PAD: Fill sequences to uniform length in batches
* SEP/CLS: Task-specific tokens for classification
* MASK: Placeholder for masked language modeling

=== Step 3: Text Encoding ===
[[step::Principle:huggingface_transformers_Text_Encoding]]

Convert raw text to token IDs using the tokenizer's vocabulary. The encoding process splits text into subwords, maps them to IDs, and adds special tokens. Attention masks indicate real vs. padding tokens.

'''Encoding outputs:'''
* input_ids: Token ID sequence
* attention_mask: Binary mask (1 for real tokens, 0 for padding)
* token_type_ids: Segment IDs for sentence pairs
* offset_mapping: Character spans for each token (fast only)

=== Step 4: Padding and Truncation ===
[[step::Principle:huggingface_transformers_Padding_Truncation]]

Handle variable-length sequences for efficient batching. Padding adds tokens to reach a target length; truncation removes tokens that exceed maximum length. Multiple strategies exist for different use cases.

'''Strategies:'''
* Padding: longest (batch max), max_length, do_not_pad
* Truncation: longest_first, only_first, only_second
* Padding side: left (for generation) or right (for classification)

=== Step 5: Chat Template Application ===
[[step::Principle:huggingface_transformers_Chat_Templates]]

Apply chat formatting templates for instruction-tuned and chat models. Templates convert conversation histories to the specific format expected by each model (ChatML, Llama, Mistral, etc.).

'''Template features:'''
* Jinja2-based templates for flexibility
* Support for system prompts, user/assistant turns
* Tool calling and function call formatting
* Generation prompts for inference

=== Step 6: Text Decoding ===
[[step::Principle:huggingface_transformers_Text_Decoding]]

Convert token IDs back to readable text. Decoding handles special token removal, subword merging, and Unicode normalization. Options control whether special tokens appear in output.

'''Decoding options:'''
* skip_special_tokens: Remove BOS, EOS, PAD from output
* clean_up_tokenization_spaces: Normalize whitespace
* Batch decoding for multiple sequences
* Streaming decode for generation

== Execution Diagram ==
{{#mermaid:graph TD
    A[Tokenizer Loading] --> B[Special Tokens Configuration]
    B --> C[Text Encoding]
    C --> D[Padding and Truncation]
    D --> E[Chat Template Application]
    E --> F[Text Decoding]
}}

== Related Pages ==
* [[step::Principle:huggingface_transformers_Tokenizer_Loading]]
* [[step::Principle:huggingface_transformers_Special_Tokens]]
* [[step::Principle:huggingface_transformers_Text_Encoding]]
* [[step::Principle:huggingface_transformers_Padding_Truncation]]
* [[step::Principle:huggingface_transformers_Chat_Templates]]
* [[step::Principle:huggingface_transformers_Text_Decoding]]
