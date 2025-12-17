{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|Transformers Docs|https://huggingface.co/docs/transformers]]
|-
! Domains
| [[domain::NLP]], [[domain::Tokenization]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

== Overview ==

Special control tokens that provide structural information and handle edge cases in text processing.

=== Description ===

Special tokens are designated vocabulary entries that serve control functions rather than representing words or subwords. They mark sequence boundaries (BOS/EOS), indicate padding, separate segments, mask tokens for training objectives, and represent unknown vocabulary. These tokens receive special treatment during tokenization (never split) and decoding (can be removed). Models learn to interpret special tokens as structural signals that guide processing and attention.

=== Usage ===

Use special tokens to structure inputs for model consumption, enable batching with padding, support sequence-pair tasks, implement masked language modeling, and handle out-of-vocabulary words gracefully. Essential for proper model input formatting and task-specific architectures.

== Theoretical Basis ==

=== Core Concepts ===

'''Standard Special Tokens:'''
* '''BOS (Beginning-of-Sequence)''': Marks sequence start, helps model initialize hidden states
* '''EOS (End-of-Sequence)''': Marks sequence end, signals generation stopping point
* '''PAD (Padding)''': Fills shorter sequences to match batch length, ignored by attention
* '''UNK (Unknown)''': Represents out-of-vocabulary words
* '''SEP (Separator)''': Separates different segments in sequence pairs
* '''CLS (Classification)''': Aggregates sequence information for classification tasks
* '''MASK''': Replaces tokens in masked language modeling objectives

'''Token Properties:'''
* '''Special Flag''': Indicates token should never be split during tokenization
* '''Normalization''': Special tokens typically bypass text normalization
* '''Attention Handling''': Some special tokens (PAD) are masked in attention
* '''ID Reservation''': Special tokens receive fixed vocabulary IDs

=== Algorithm ===

<syntaxhighlight lang="text">
function ADD_SPECIAL_TOKENS(tokenizer, special_tokens_dict):
    tokens_to_add = []
    num_added = 0

    for token_name, token_value in special_tokens_dict:
        // Validate token name
        if token_name not in ALLOWED_SPECIAL_TOKENS:
            raise ValueError("Invalid special token name")

        // Convert string to AddedToken object
        if is_string(token_value):
            token_obj = AddedToken(
                content=token_value,
                special=True,
                normalized=False,
                lstrip=False,
                rstrip=False
            )
        else:
            token_obj = token_value

        // Check if token already exists
        if token_value in tokenizer.vocabulary:
            // Update attribute but don't add to vocab
            setattr(tokenizer, token_name, token_obj)
        else:
            // Add to vocabulary
            token_id = len(tokenizer.vocabulary)
            tokenizer.vocabulary[token_value] = token_id
            setattr(tokenizer, token_name, token_obj)
            num_added += 1

        // Register for special handling
        tokenizer.special_tokens_set.add(token_value)

    return num_added

function APPLY_SPECIAL_TOKENS(tokenizer, token_ids, add_special_tokens=True):
    if not add_special_tokens:
        return token_ids

    // Build sequence with model-specific special tokens
    result = []

    if tokenizer.bos_token_id is not None:
        result.append(tokenizer.bos_token_id)

    if tokenizer.cls_token_id is not None:
        result.append(tokenizer.cls_token_id)

    result.extend(token_ids)

    if tokenizer.eos_token_id is not None:
        result.append(tokenizer.eos_token_id)

    if tokenizer.sep_token_id is not None:
        result.append(tokenizer.sep_token_id)

    return result
</syntaxhighlight>

=== Key Properties ===

* '''Atomicity''': Special tokens are never split or decomposed during tokenization
* '''Universality''': Common special tokens (PAD, BOS, EOS) exist across most tokenizers
* '''Model-Specificity''': Different architectures use different special token conventions
* '''Embedding Space''': Special tokens have learnable embeddings like regular tokens
* '''Skip-ability''': Special tokens can be filtered during decoding

=== Model-Specific Conventions ===

'''BERT-style:'''
* [CLS] at start for classification
* [SEP] between segments and at end
* [MASK] for masked tokens
* [PAD] for padding
* [UNK] for unknown words

'''GPT-style:'''
* <|endoftext|> for BOS/EOS/PAD (same token)
* No segment separators (decoder-only)
* <|im_start|>, <|im_end|> for instruction tuning

'''T5-style:'''
* </s> for EOS
* <pad> for padding
* Task prefixes as pseudo-special tokens
* <extra_id_X> for span masking

=== Design Considerations ===

* '''Vocabulary Size Impact''': Each special token consumes one vocabulary slot
* '''Embedding Initialization''': Special tokens need careful initialization during fine-tuning
* '''Model Compatibility''': Adding tokens requires resizing model embeddings
* '''Tokenization Ambiguity''': Special tokens in raw text must be handled carefully

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_transformers_add_special_tokens]]
