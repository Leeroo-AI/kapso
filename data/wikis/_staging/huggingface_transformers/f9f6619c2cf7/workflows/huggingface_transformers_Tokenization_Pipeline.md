{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Tokenizer Guide|https://huggingface.co/docs/transformers/main_classes/tokenizer]]
* [[source::Doc|Fast Tokenizers|https://huggingface.co/docs/transformers/fast_tokenizers]]
|-
! Domains
| [[domain::NLP]], [[domain::Tokenization]], [[domain::Text_Processing]], [[domain::Encoding]]
|-
! Last Updated
| [[last_updated::2025-12-18 14:00 GMT]]
|}

== Overview ==
End-to-end text tokenization workflow that converts raw text into model-ready token sequences, supporting both Python-based slow tokenizers and Rust-based fast tokenizers.

=== Description ===
The tokenization system in HuggingFace Transformers provides a unified interface for encoding text into token IDs and decoding token IDs back to text. It supports multiple backends:

* **Python Tokenizers**: Native Python implementations for various tokenization algorithms
* **Rust Tokenizers**: High-performance tokenizers via the `tokenizers` library
* **SentencePiece**: Integration with Google's SentencePiece library
* **Tiktoken**: Support for OpenAI's tiktoken encoding

Key features include:
* Vocabulary management and special token handling
* Padding and truncation strategies
* Batch encoding with attention masks
* Chat templates for conversational models
* Fast → Slow conversion utilities

=== Usage ===
Execute this workflow when you need to:
* Convert text to token IDs for model input
* Decode model outputs back to readable text
* Handle special tokens (BOS, EOS, PAD, etc.)
* Process text in batches with consistent padding
* Apply chat templates for instruction-tuned models

Prerequisites:
* Tokenizer checkpoint (from Hub or local)
* Raw text input(s)

== Execution Steps ==

=== Step 1: Tokenizer Loading ===
[[step::Principle:huggingface_transformers_Tokenizer_Loading]]

Load the tokenizer from a checkpoint. AutoTokenizer automatically selects the appropriate tokenizer class based on the model configuration. Both slow (Python) and fast (Rust) tokenizer variants are supported.

'''Loading behavior:'''
* Detect tokenizer class from tokenizer_config.json
* Load vocabulary files (vocab.json, merges.txt, tokenizer.json)
* Initialize special tokens from config
* Prefer fast tokenizer if available (use_fast=True by default)

=== Step 2: Vocabulary Initialization ===
[[step::Principle:huggingface_transformers_Vocabulary_Initialization]]

Initialize the tokenizer's vocabulary and merges. Different tokenization algorithms have different vocabulary structures (BPE merges, WordPiece vocab, Unigram model).

'''Vocabulary components:'''
* Token-to-ID mappings
* Special tokens mapping (PAD, UNK, CLS, SEP, etc.)
* Merge rules for subword algorithms
* Added tokens for custom vocabulary

=== Step 3: Text Normalization ===
[[step::Principle:huggingface_transformers_Text_Normalization]]

Apply text normalization before tokenization. This may include Unicode normalization, lowercasing, whitespace handling, and model-specific preprocessing.

'''Normalization operations:'''
* Unicode NFC/NFKC normalization
* Accent stripping
* Lowercasing (if configured)
* Whitespace normalization

=== Step 4: Pre-Tokenization ===
[[step::Principle:huggingface_transformers_Pre_Tokenization]]

Split text into preliminary tokens before subword segmentation. Pre-tokenization rules vary by model (whitespace splitting, punctuation handling, digit splitting).

'''Pre-tokenization strategies:'''
* Whitespace splitting
* Punctuation isolation
* ByteLevel for GPT-2 style
* Metaspace for SentencePiece models

=== Step 5: Subword Tokenization ===
[[step::Principle:huggingface_transformers_Subword_Tokenization]]

Apply the core tokenization algorithm to convert pre-tokens into subword tokens. Support for BPE, WordPiece, Unigram, and SentencePiece algorithms.

'''Algorithms:'''
* BPE (Byte-Pair Encoding): Iterative merge-based
* WordPiece: Greedy longest-match
* Unigram: Probabilistic subword selection
* SentencePiece: Language-agnostic subword

=== Step 6: Token ID Conversion ===
[[step::Principle:huggingface_transformers_Token_ID_Conversion]]

Convert tokenized subwords to integer IDs using the vocabulary. Handle unknown tokens and special token insertion.

'''Conversion process:'''
* Map tokens to vocabulary IDs
* Handle OOV with UNK token
* Insert special tokens (CLS, SEP, etc.)
* Build token type IDs if needed

=== Step 7: Padding and Truncation ===
[[step::Principle:huggingface_transformers_Padding_Truncation]]

Apply padding and truncation to create fixed-length sequences suitable for batched model input. Generate attention masks to indicate real vs. padded tokens.

'''Padding strategies:'''
* Left padding (for generation)
* Right padding (for classification)
* Max length truncation
* Stride for long documents

=== Step 8: Output Encoding Creation ===
[[step::Principle:huggingface_transformers_Encoding_Creation]]

Assemble the final BatchEncoding with all required tensors. Include input_ids, attention_mask, token_type_ids, and optional offset mappings.

'''Output contents:'''
* input_ids: Token ID sequences
* attention_mask: 1 for real tokens, 0 for padding
* token_type_ids: Segment IDs for sequence pairs
* offset_mapping: Character spans (fast tokenizer only)

== Execution Diagram ==
{{#mermaid:graph TD
    A[Tokenizer Loading] --> B[Vocabulary Initialization]
    B --> C[Text Normalization]
    C --> D[Pre-Tokenization]
    D --> E[Subword Tokenization]
    E --> F[Token ID Conversion]
    F --> G[Padding & Truncation]
    G --> H[Output Encoding Creation]
}}

== Related Pages ==
* [[step::Principle:huggingface_transformers_Tokenizer_Loading]]
* [[step::Principle:huggingface_transformers_Vocabulary_Initialization]]
* [[step::Principle:huggingface_transformers_Text_Normalization]]
* [[step::Principle:huggingface_transformers_Pre_Tokenization]]
* [[step::Principle:huggingface_transformers_Subword_Tokenization]]
* [[step::Principle:huggingface_transformers_Token_ID_Conversion]]
* [[step::Principle:huggingface_transformers_Padding_Truncation]]
* [[step::Principle:huggingface_transformers_Encoding_Creation]]
