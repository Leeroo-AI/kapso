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

Converting human-readable text into numerical token sequences that neural networks can process.

=== Description ===

Text encoding transforms raw text strings into fixed-size integer sequences by splitting text into tokens (words, subwords, or characters) and mapping each token to a unique vocabulary ID. The process includes text normalization, tokenization algorithm application (BPE, WordPiece, Unigram), special token insertion, sequence length management, and generation of auxiliary inputs (attention masks, token type IDs). The encoded output provides a standardized numerical representation compatible with model architectures.

=== Usage ===

Use text encoding as the mandatory first step before feeding text into neural models. Required for all text-based tasks including classification, generation, translation, question answering, and sequence labeling. Ensures text is in the exact format expected by pre-trained models.

== Theoretical Basis ==

=== Core Concepts ===

'''Encoding Pipeline:'''
1. '''Pre-normalization''': Unicode normalization, lowercasing, accent removal
2. '''Pre-tokenization''': Splitting on whitespace and punctuation
3. '''Tokenization''': Applying algorithm (BPE/WordPiece/Unigram) to split into subwords
4. '''Post-processing''': Adding special tokens, creating attention masks
5. '''Numericalization''': Converting tokens to vocabulary IDs

'''Key Components:'''
* '''input_ids''': Core sequence of token IDs
* '''attention_mask''': Binary mask (1=real token, 0=padding)
* '''token_type_ids''': Segment IDs for sequence pairs (0=first, 1=second)

'''Encoding Algorithms:'''
* '''Byte-Pair Encoding (BPE)''': Iteratively merges most frequent character pairs
* '''WordPiece''': Similar to BPE but uses likelihood maximization
* '''Unigram''': Starts with large vocabulary, iteratively removes tokens
* '''Character-level''': Each character is a token (no OOV)

=== Algorithm ===

<syntaxhighlight lang="text">
function ENCODE_TEXT(tokenizer, text, options):
    // Step 1: Pre-normalization
    if tokenizer.do_lower_case:
        text = text.lower()
    text = normalize_unicode(text)

    // Step 2: Pre-tokenization (split into words)
    words = pre_tokenize(text)  // ["Hello", ",", "world"]

    // Step 3: Apply tokenization algorithm
    all_tokens = []
    for word in words:
        if word in tokenizer.vocabulary:
            all_tokens.append(word)
        else:
            // Apply subword tokenization
            subwords = tokenizer.tokenize_word(word)
            all_tokens.extend(subwords)

    // Step 4: Handle unknown tokens
    tokens = []
    for token in all_tokens:
        if token in tokenizer.vocabulary:
            tokens.append(token)
        else:
            tokens.append(tokenizer.unk_token)

    // Step 5: Convert tokens to IDs
    token_ids = [tokenizer.token_to_id[t] for t in tokens]

    // Step 6: Add special tokens
    if options.add_special_tokens:
        token_ids = [tokenizer.cls_token_id] + token_ids + [tokenizer.sep_token_id]

    // Step 7: Handle truncation
    if options.truncation and len(token_ids) > options.max_length:
        token_ids = token_ids[:options.max_length]
        token_ids[-1] = tokenizer.sep_token_id  // Ensure EOS

    // Step 8: Create attention mask
    attention_mask = [1] * len(token_ids)

    // Step 9: Handle padding
    if options.padding == "max_length":
        pad_length = options.max_length - len(token_ids)
        token_ids.extend([tokenizer.pad_token_id] * pad_length)
        attention_mask.extend([0] * pad_length)

    return {
        "input_ids": token_ids,
        "attention_mask": attention_mask
    }

function ENCODE_SEQUENCE_PAIR(tokenizer, text_a, text_b, options):
    // Encode both sequences
    ids_a = encode_without_special_tokens(text_a)
    ids_b = encode_without_special_tokens(text_b)

    // Build sequence with separators
    token_ids = [tokenizer.cls_token_id] + ids_a + [tokenizer.sep_token_id] + ids_b + [tokenizer.sep_token_id]

    // Create token type IDs
    token_type_ids = [0] * (len(ids_a) + 2) + [1] * (len(ids_b) + 1)

    // Handle truncation (smart truncation of longer sequence)
    if len(token_ids) > options.max_length:
        if len(ids_a) > len(ids_b):
            overflow = len(token_ids) - options.max_length
            ids_a = ids_a[:len(ids_a) - overflow]
        else:
            overflow = len(token_ids) - options.max_length
            ids_b = ids_b[:len(ids_b) - overflow]
        // Rebuild
        token_ids = [tokenizer.cls_token_id] + ids_a + [tokenizer.sep_token_id] + ids_b + [tokenizer.sep_token_id]
        token_type_ids = [0] * (len(ids_a) + 2) + [1] * (len(ids_b) + 1)

    return {
        "input_ids": token_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": [1] * len(token_ids)
    }
</syntaxhighlight>

=== Key Properties ===

* '''Determinism''': Same text always produces same token sequence
* '''Reversibility''': Tokens can be decoded back to approximate original text
* '''Length Sensitivity''': Output length depends on input text and tokenization algorithm
* '''Vocabulary Coverage''': All text can be encoded (via UNK token if needed)
* '''Model Compatibility''': Encoding must match model's training tokenization

=== Mathematical Formulation ===

Given text <math>T = (c_1, c_2, \ldots, c_n)</math> where <math>c_i</math> are characters:

1. '''Tokenization:''' <math>T \rightarrow (t_1, t_2, \ldots, t_m)</math> where <math>t_j \in V</math> (vocabulary)

2. '''ID Mapping:''' <math>E(t_j) = \text{vocab}[t_j] \in \mathbb{N}</math>

3. '''Sequence:''' <math>S = (E(t_1), E(t_2), \ldots, E(t_m))</math>

4. '''With Special Tokens:''' <math>S' = ([CLS], E(t_1), \ldots, E(t_m), [SEP])</math>

5. '''Attention Mask:''' <math>M = (1, 1, \ldots, 1, 0, \ldots, 0)</math> where 1s cover real tokens

=== Design Considerations ===

* '''Batch Processing''': Efficient encoding of multiple texts simultaneously
* '''Memory Efficiency''': Large vocabularies require efficient storage
* '''Speed Optimization''': Fast tokenizers (Rust-based) provide 10-100x speedup
* '''Unicode Handling''': Proper handling of multi-byte characters and emoji
* '''Streaming Support''': Ability to encode text incrementally

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_transformers_tokenizer_call]]
