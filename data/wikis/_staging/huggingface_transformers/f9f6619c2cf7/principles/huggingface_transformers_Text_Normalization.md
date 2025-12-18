{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|Transformers Tokenizers|https://huggingface.co/docs/transformers/main_classes/tokenizer]]
* [[source::Doc|HuggingFace Tokenizers Library|https://huggingface.co/docs/tokenizers/api/normalizers]]
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
|-
! Domains
| [[domain::NLP]], [[domain::Preprocessing]]
|-
! Last Updated
| [[last_updated::2025-12-18 00:00 GMT]]
|}

== Overview ==
Applying a sequence of text transformations to standardize input strings before tokenization, ensuring consistent vocabulary matching.

=== Description ===
Text normalization is the first stage of text processing in the tokenization pipeline, converting raw input text into a standardized form. This principle addresses the problem of text variation (different Unicode representations, case differences, accents) that would otherwise cause the same semantic content to map to different tokens. It fits at the very beginning of the tokenization process, before any token splitting occurs.

Normalization operations include Unicode normalization (NFD/NFC/NFKD/NFKC), lowercasing, accent stripping, whitespace normalization, and character filtering. The specific sequence and types of normalizations depend on how the model was trained. Proper normalization ensures that input text matches the format seen during model training, which is critical for vocabulary lookup and model performance.

=== Usage ===
This principle should be applied when:
* Processing raw text before tokenization
* Standardizing text for vocabulary matching
* Implementing model-specific preprocessing requirements
* Handling multilingual text with various Unicode forms
* Ensuring consistency between training and inference preprocessing

== Theoretical Basis ==
The text normalization principle follows these logical steps:

1. '''Unicode Normalization''': Standardize Unicode representation
   * NFD (Canonical Decomposition): Split accented characters into base + combining marks
   * NFC (Canonical Composition): Combine base + marks into precomposed forms
   * NFKD (Compatibility Decomposition): Apply compatibility equivalences
   * NFKC (Compatibility Composition): Compose after compatibility decomposition
   * Purpose: Ensure "e" + combining accent equals precomposed "é"

2. '''Case Normalization''': Standardize character case
   * Lowercase: Convert all characters to lowercase
   * Uppercase: Convert all characters to uppercase (rare)
   * Casefolding: More aggressive case standardization for matching
   * Purpose: Make "Hello" and "hello" equivalent

3. '''Accent Stripping''': Remove diacritical marks
   * After NFD normalization, remove combining mark characters
   * Transform "café" → "cafe", "naïve" → "naive"
   * Purpose: Reduce vocabulary size for accent-insensitive models

4. '''Whitespace Normalization''': Standardize spacing
   * Replace multiple spaces with single space
   * Normalize tab, newline, and other whitespace to spaces
   * Trim leading and trailing whitespace
   * Purpose: Consistent token boundaries

5. '''Character Filtering''': Remove or replace unwanted characters
   * Strip control characters
   * Replace or remove specific Unicode categories
   * Handle zero-width characters
   * Purpose: Clean text of problematic characters

6. '''Normalizer Sequencing''': Apply normalizations in order
   * Order matters: lowercase after accent stripping gives different results
   * Compose normalizers into processing pipeline
   * Cache normalized results for efficiency
   * Purpose: Consistent multi-stage normalization

Pseudocode:
```
function normalize_text(text, normalizer_sequence):
    normalized = text
    for normalizer in normalizer_sequence:
        normalized = normalizer.normalize(normalized)
    return normalized

function create_bert_normalizer():
    return Sequence([
        NFD(),           # Decompose accented characters
        Lowercase(),     # Convert to lowercase
        StripAccents()   # Remove combining marks
    ])

function normalize_for_tokenization(text, tokenizer):
    if tokenizer.normalizer is not None:
        return tokenizer.normalizer.normalize(text)
    return text
```

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_transformers_Normalizer_application]]
