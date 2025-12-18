{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|Transformers Tokenizers|https://huggingface.co/docs/transformers/main_classes/tokenizer]]
* [[source::Doc|HuggingFace Tokenizers Library|https://huggingface.co/docs/tokenizers/api/pre-tokenizers]]
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
|-
! Domains
| [[domain::NLP]], [[domain::Preprocessing]]
|-
! Last Updated
| [[last_updated::2025-12-18 00:00 GMT]]
|}

== Overview ==
Splitting normalized text into preliminary token units before applying subword tokenization algorithms.

=== Description ===
Pre-tokenization is the second stage of text processing in the tokenization pipeline, dividing normalized text into coarse token boundaries that the subword tokenization algorithm will further split if needed. This principle solves the problem of defining initial segmentation boundaries (e.g., at whitespace or punctuation) that respect linguistic structure while enabling efficient subword tokenization. It fits between text normalization and subword tokenization, establishing token candidates that won't be merged across boundaries.

Pre-tokenization strategies vary by model architecture and training approach. Common strategies include whitespace splitting (GPT-2), whitespace + punctuation splitting (BERT), byte-level pre-tokenization (GPT-2, RoBERTa), character splitting, and no pre-tokenization (some character-based models). The choice of pre-tokenizer determines how tokens can be combined and affects the resulting vocabulary structure.

=== Usage ===
This principle should be applied when:
* Processing normalized text before subword tokenization
* Establishing initial token boundaries that won't be crossed
* Implementing model-specific segmentation requirements
* Handling different writing systems (space-delimited vs non-space-delimited)
* Preserving specific text structures (URLs, email addresses, numbers)

== Theoretical Basis ==
The pre-tokenization principle follows these logical steps:

1. '''Whitespace Splitting''': Segment at space boundaries
   * Split text at space characters (space, tab, newline)
   * Preserve non-space characters within tokens
   * Result: ["Hello", "world", "!"]
   * Purpose: Respect word boundaries in space-delimited languages

2. '''Punctuation Handling''': Isolate punctuation marks
   * Split before and after punctuation characters
   * Treat punctuation as separate tokens
   * Example: "Hello, world!" → ["Hello", ",", "world", "!"]
   * Purpose: Allow punctuation to have independent embeddings

3. '''Byte-Level Pre-tokenization''': Character-to-byte mapping
   * Map all Unicode characters to valid UTF-8 bytes
   * Ensure every byte has a unique token representation
   * Avoids unknown tokens for any Unicode character
   * Purpose: Handle arbitrary Unicode without UNK tokens

4. '''Digit Splitting''': Separate individual digits
   * Split multi-digit numbers into individual digits
   * Example: "2023" → ["2", "0", "2", "3"]
   * Purpose: Better generalization for numeric values

5. '''Boundary Preservation''': Maintain split points
   * Pre-tokenization boundaries are never crossed by subword algorithm
   * If pre-tokenization produces ["Hello", "world"], subword tokenization won't create "lloworl"
   * Each pre-token is processed independently
   * Purpose: Enforce linguistic segmentation constraints

6. '''Offset Tracking''': Maintain character positions
   * Track original character offsets for each pre-token
   * Enable mapping tokens back to source text spans
   * Support for extractive QA and NER applications
   * Purpose: Preserve alignment with original text

Pseudocode:
```
function pre_tokenize_whitespace(text):
    pre_tokens = []
    for word in text.split():
        pre_tokens.append(word)
    return pre_tokens

function pre_tokenize_whitespace_punctuation(text):
    pre_tokens = []
    current_token = ""

    for char in text:
        if is_whitespace(char):
            if current_token:
                pre_tokens.append(current_token)
                current_token = ""
        elif is_punctuation(char):
            if current_token:
                pre_tokens.append(current_token)
            pre_tokens.append(char)
            current_token = ""
        else:
            current_token += char

    if current_token:
        pre_tokens.append(current_token)

    return pre_tokens

function pre_tokenize_byte_level(text):
    # Convert Unicode to byte representation
    byte_text = []
    for char in text:
        bytes = utf8_encode(char)
        for byte in bytes:
            byte_token = byte_to_unicode_mapping[byte]
            byte_text.append(byte_token)
    return byte_text

function apply_pre_tokenization(text, pre_tokenizer):
    normalized = normalize(text)
    pre_tokens = pre_tokenizer.pre_tokenize(normalized)
    return pre_tokens
```

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_transformers_PreTokenizer_application]]
