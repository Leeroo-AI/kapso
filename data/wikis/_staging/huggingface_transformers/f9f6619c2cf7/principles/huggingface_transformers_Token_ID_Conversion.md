{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|Transformers Tokenizers|https://huggingface.co/docs/transformers/main_classes/tokenizer]]
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
|-
! Domains
| [[domain::NLP]], [[domain::Preprocessing]]
|-
! Last Updated
| [[last_updated::2025-12-18 00:00 GMT]]
|}

== Overview ==
Bidirectional mapping between token strings and integer IDs using vocabulary dictionaries.

=== Description ===
Token ID conversion is the fundamental operation that enables translation between human-readable token strings and machine-processable integer IDs. This principle solves the problem of representing discrete vocabulary items in a format suitable for neural network processing (integers for embedding lookups) while maintaining the ability to convert back to text for interpretation. It's a utility operation used throughout the tokenization pipeline and during decoding.

The conversion uses bidirectional dictionaries maintained by the tokenizer: token-to-ID mapping (vocab) and ID-to-token mapping (inverse_vocab). The process handles both base vocabulary tokens and added tokens (including special tokens), ensures consistent lookups with unknown token handling, and supports both single token and batch conversions. This enables seamless translation between the string and numeric representations of tokens.

=== Usage ===
This principle should be applied when:
* Looking up token IDs during encoding after subword segmentation
* Converting IDs back to tokens during decoding
* Inspecting or debugging tokenization outputs
* Implementing custom tokenization logic
* Analyzing model vocabulary composition

== Theoretical Basis ==
The token ID conversion principle follows these logical steps:

1. '''Vocabulary Structure''': Maintain bidirectional mappings
   * '''token_to_id dictionary''': Maps token strings to integer IDs
     - Example: {"hello": 1000, "world": 2000, "[CLS]": 101}
     - Includes base vocabulary + added tokens
   * '''id_to_token dictionary''': Maps integer IDs to token strings
     - Example: {1000: "hello", 2000: "world", 101: "[CLS]"}
     - Inverse of token_to_id for decoding

2. '''Token to ID Conversion''': String to integer lookup
   * For single token string:
     - Look up token in token_to_id dictionary
     - If found, return corresponding ID
     - If not found, return unknown token ID (unk_token_id)
   * For list of token strings:
     - Apply conversion to each token
     - Return list of IDs

3. '''ID to Token Conversion''': Integer to string lookup
   * For single ID:
     - Look up ID in id_to_token dictionary
     - If found, return corresponding token string
     - If not found, return unknown token string (unk_token)
   * For list of IDs:
     - Apply conversion to each ID
     - Return list of token strings

4. '''Added Token Handling''': Prioritize added tokens
   * Added tokens (special tokens, user-added tokens) may override base vocabulary
   * Check added_tokens_encoder first, then base vocabulary
   * Ensures special tokens like [CLS], [PAD] have consistent IDs
   * Added tokens may have special properties (normalization, stripping)

5. '''Unknown Token Fallback''': Handle missing entries
   * If token not in vocabulary during encoding:
     - Return unk_token_id (usually 0, 1, or 100 depending on tokenizer)
   * If ID not in vocabulary during decoding:
     - Return unk_token string (usually "[UNK]", "<unk>", or "")
   * Prevents errors from missing vocabulary entries

6. '''Special Token Recognition''': Identify special token IDs
   * Maintain set of all_special_ids for quick membership testing
   * Used for:
     - Skipping special tokens during decoding
     - Creating special token masks
     - Filtering out special tokens from outputs
   * Examples: CLS, SEP, PAD, MASK, BOS, EOS tokens

Pseudocode:
```
function convert_tokens_to_ids(tokens, tokenizer):
    if isinstance(tokens, str):
        # Single token
        return tokenizer.vocab.get(tokens, tokenizer.unk_token_id)
    else:
        # List of tokens
        return [tokenizer.vocab.get(token, tokenizer.unk_token_id) for token in tokens]

function convert_ids_to_tokens(ids, tokenizer, skip_special_tokens=False):
    if isinstance(ids, int):
        # Single ID
        if skip_special_tokens and ids in tokenizer.all_special_ids:
            return None
        return tokenizer.inverse_vocab.get(ids, tokenizer.unk_token)
    else:
        # List of IDs
        tokens = []
        for id in ids:
            if skip_special_tokens and id in tokenizer.all_special_ids:
                continue
            tokens.append(tokenizer.inverse_vocab.get(id, tokenizer.unk_token))
        return tokens

function get_vocab(tokenizer):
    # Return complete vocabulary mapping
    vocab = dict(tokenizer.base_vocab)  # Start with base vocabulary

    # Add special tokens and added tokens
    vocab.update(tokenizer.added_tokens_encoder)

    return vocab
```

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_transformers_Convert_tokens_to_ids]]
