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
Converting text into sequences of token IDs by applying subword segmentation algorithms and vocabulary lookup.

=== Description ===
Subword tokenization is the core encoding step that transforms pre-tokenized text into integer sequences that neural networks can process. This principle addresses the problem of balancing vocabulary size with coverage: whole-word vocabularies are too large and suffer from out-of-vocabulary issues, while character-level vocabularies are too granular and create very long sequences. Subword tokenization fits after pre-tokenization and before padding/truncation, performing the actual vocabulary lookup and ID assignment.

The encoding process applies a subword algorithm (BPE, WordPiece, Unigram, SentencePiece) to split pre-tokens into subword units, looks up each subword in the vocabulary to get its ID, adds special tokens (CLS, SEP, BOS, EOS) according to model requirements, and returns a list of token IDs. This enables models to handle rare words by decomposing them into known subwords while keeping common words as single tokens.

=== Usage ===
This principle should be applied when:
* Converting text to model input (inference or training)
* Encoding single sequences or sequence pairs
* Preparing text for embedding lookup in neural networks
* Handling out-of-vocabulary words through subword decomposition
* Creating inputs that match model training format

== Theoretical Basis ==
The subword tokenization principle follows these logical steps:

1. '''Subword Segmentation''': Split pre-tokens into subword units
   * '''BPE (Byte Pair Encoding)''': Iteratively merge most frequent byte/character pairs
     - Start with character-level tokens
     - Merge pairs based on learned merge rules
     - Example: "lower" → ["low", "er"]
   * '''WordPiece''': Similar to BPE but uses likelihood maximization
     - Adds "##" prefix to non-initial subwords
     - Example: "lower" → ["low", "##er"]
   * '''Unigram''': Probabilistic subword segmentation
     - Multiple possible segmentations with probabilities
     - Choose highest probability segmentation
   * '''SentencePiece''': Treats text as Unicode, no pre-tokenization required
     - Integrated normalization and subword segmentation

2. '''Vocabulary Lookup''': Convert subword strings to IDs
   * For each subword produced by segmentation:
     - Look up subword in vocabulary dictionary
     - Retrieve corresponding integer ID
     - Handle unknown tokens (map to UNK token ID)
   * Result: List of token IDs representing the text

3. '''Special Token Addition''': Add model-specific special tokens
   * '''BERT-style (masked language modeling)''':
     - Single sequence: [CLS] tokens [SEP]
     - Sequence pair: [CLS] seq1 [SEP] seq2 [SEP]
   * '''GPT-style (causal language modeling)''':
     - Single sequence: tokens <|endoftext|>
     - No CLS token, just EOS at end
   * '''T5-style (encoder-decoder)''':
     - Encoder: tokens </s>
     - Decoder: <pad> tokens </s>
   * '''RoBERTa-style''':
     - Single sequence: <s> tokens </s>
     - Sequence pair: <s> seq1 </s></s> seq2 </s>

4. '''Segment ID Generation''': Create token type IDs for sequence pairs
   * First sequence tokens: segment ID = 0
   * Second sequence tokens: segment ID = 1
   * Special tokens: typically segment ID = 0 (model-dependent)
   * Used by BERT-family models to distinguish sequences

5. '''Attention Mask Creation''': Generate mask for valid tokens
   * Real tokens: mask value = 1
   * Padding tokens: mask value = 0
   * Prevents attention to padding in transformers

6. '''Length Handling''': Apply truncation if needed
   * If sequence exceeds max_length:
     - Truncate from right (default) or left
     - For sequence pairs, truncate longest first or both equally
     - Preserve special tokens after truncation
   * If add_special_tokens=True, account for their length

Pseudocode:
```
function encode_text(text, tokenizer, add_special_tokens=True, max_length=None):
    # Step 1: Normalize
    normalized = tokenizer.normalizer.normalize(text)

    # Step 2: Pre-tokenize
    pre_tokens = tokenizer.pre_tokenizer.pre_tokenize(normalized)

    # Step 3: Subword tokenization
    token_ids = []
    for pre_token in pre_tokens:
        subwords = tokenizer.model.tokenize(pre_token)
        for subword in subwords:
            token_id = tokenizer.vocab.get(subword, tokenizer.unk_token_id)
            token_ids.append(token_id)

    # Step 4: Add special tokens
    if add_special_tokens:
        token_ids = tokenizer.add_special_tokens(token_ids)

    # Step 5: Truncation
    if max_length and len(token_ids) > max_length:
        token_ids = token_ids[:max_length]

    return token_ids

function encode_pair(text1, text2, tokenizer):
    ids1 = encode_without_special_tokens(text1, tokenizer)
    ids2 = encode_without_special_tokens(text2, tokenizer)

    # BERT-style: [CLS] text1 [SEP] text2 [SEP]
    token_ids = [CLS_ID] + ids1 + [SEP_ID] + ids2 + [SEP_ID]
    segment_ids = [0] * (len(ids1) + 2) + [1] * (len(ids2) + 1)

    return token_ids, segment_ids
```

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_transformers_Tokenizer_encode]]
