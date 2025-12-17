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

Converting numerical token sequences back into human-readable text strings.

=== Description ===

Text decoding is the inverse operation of encoding, transforming token ID sequences back into text. The process involves mapping IDs to tokens via vocabulary lookup, handling subword token merging (e.g., removing "##" prefixes in WordPiece, joining byte-pair pieces), optionally filtering special tokens, and applying post-processing cleanup (space normalization, punctuation handling). Decoding must handle various tokenization artifacts while producing natural, readable text that approximates the original input or represents model-generated content.

=== Usage ===

Use text decoding to interpret model outputs, display generated text, evaluate predictions, and convert any tokenized representation back to human-readable form. Essential for all generation tasks (translation, summarization, chat), model debugging, and presenting results to users.

== Theoretical Basis ==

=== Core Concepts ===

'''Decoding Pipeline:'''
1. '''ID to Token Mapping''': Look up each token ID in vocabulary
2. '''Subword Merging''': Combine subword pieces into words
3. '''Special Token Filtering''': Optionally remove special tokens
4. '''Byte-to-Character Conversion''': For byte-level BPE (GPT-2 style)
5. '''Space Normalization''': Clean up spacing artifacts
6. '''Punctuation Handling''': Fix spacing around punctuation

'''Token Representation Types:'''
* '''Word-level''': Direct word to token mapping (no merging needed)
* '''Subword-level''': Tokens are word pieces requiring merging
* '''Character-level''': Each character is a token (trivial merging)
* '''Byte-level''': Bytes mapped to tokens (UTF-8 decoding required)

'''Merging Strategies:'''
* '''WordPiece (BERT)''': Remove "##" continuation markers, join pieces
* '''BPE (GPT-2)''': Map bytes to characters, join at space boundaries
* '''SentencePiece''': Remove "▁" (underscore) space markers
* '''Unigram''': Similar to SentencePiece

=== Algorithm ===

<syntaxhighlight lang="text">
function DECODE(tokenizer, token_ids, skip_special_tokens=False):
    // Step 1: Filter special tokens if requested
    if skip_special_tokens:
        filtered_ids = []
        for id in token_ids:
            if id not in tokenizer.all_special_ids:
                filtered_ids.append(id)
        token_ids = filtered_ids

    // Step 2: Convert IDs to tokens
    tokens = []
    for id in token_ids:
        if id in tokenizer.id_to_token:
            tokens.append(tokenizer.id_to_token[id])
        else:
            tokens.append(tokenizer.unk_token)

    // Step 3: Apply tokenization-specific merging
    text = merge_tokens(tokenizer, tokens)

    // Step 4: Post-processing cleanup
    if tokenizer.clean_up_tokenization_spaces:
        text = cleanup_spaces(text)

    return text

function MERGE_TOKENS_WORDPIECE(tokens):
    // BERT-style WordPiece
    result = []
    current_word = ""

    for token in tokens:
        if token.startswith("##"):
            // Continuation of previous word
            current_word += token[2:]  // Remove "##"
        else:
            if current_word:
                result.append(current_word)
            current_word = token

    if current_word:
        result.append(current_word)

    return " ".join(result)

function MERGE_TOKENS_BPE(tokens):
    // GPT-2 style BPE
    // Tokens are bytes, need UTF-8 decoding
    text = ""
    for token in tokens:
        if token in tokenizer.byte_decoder:
            text += tokenizer.byte_decoder[token]
        else:
            text += token

    // Convert bytes to UTF-8 string
    try:
        text = bytes(text, 'utf-8').decode('utf-8', errors='ignore')
    except:
        pass

    return text

function MERGE_TOKENS_SENTENCEPIECE(tokens):
    // SentencePiece style
    text = ""
    for token in tokens:
        if token.startswith("▁"):
            text += " " + token[1:]  // Replace underscore with space
        else:
            text += token

    return text.strip()

function CLEANUP_SPACES(text):
    // Remove extra spaces
    text = re.sub(r' +', ' ', text)

    // Fix spacing around punctuation
    text = re.sub(r' ([.,!?;:])', r'\1', text)
    text = re.sub(r'([.,!?;:]) ', r'\1 ', text)

    // Remove space before apostrophes
    text = re.sub(r" '", "'", text)

    // Remove leading/trailing whitespace
    text = text.strip()

    return text

function BATCH_DECODE(tokenizer, batch_token_ids, skip_special_tokens=False):
    // Decode batch of sequences
    results = []
    for token_ids in batch_token_ids:
        text = decode(tokenizer, token_ids, skip_special_tokens)
        results.append(text)
    return results
</syntaxhighlight>

=== Mathematical Formulation ===

'''Decoding Function:'''

Given token ID sequence <math>S = (id_1, id_2, \ldots, id_n)</math>:

1. '''Token Lookup:''' <math>T = (t_1, t_2, \ldots, t_n)</math> where <math>t_i = V^{-1}(id_i)</math>

2. '''Merge Operation:''' <math>M: T \rightarrow W</math> where <math>W = (w_1, w_2, \ldots, w_m)</math>, <math>m \leq n</math>

3. '''String Join:''' <math>text = w_1 \oplus sep \oplus w_2 \oplus \ldots \oplus w_m</math> where <math>\oplus</math> is concatenation

4. '''Special Token Filter:''' <math>S' = \{id \in S : id \notin \mathcal{S}\}</math> where <math>\mathcal{S}</math> is special token set

'''Inverse Property (Approximate):'''

For encoding <math>E</math> and decoding <math>D</math>:
* <math>D(E(text)) \approx text</math> (not exact due to normalization, subwords, UNK)
* <math>E(D(ids)) = ids</math> only if ids are valid tokens

=== Key Properties ===

* '''Surjectivity''': Multiple token sequences can decode to same text
* '''Information Loss''': Cannot perfectly recover original text (normalization, OOV)
* '''Monotonicity''': Longer token sequences → longer or equal text
* '''Special Token Sensitivity''': Including/excluding specials changes output significantly
* '''Encoding Compatibility''': Decoding must match encoding tokenization algorithm

=== Edge Cases ===

'''Unknown Tokens:'''
* Token IDs not in vocabulary decode to <UNK> or raise error
* Model-generated invalid IDs need graceful handling

'''Incomplete Sequences:'''
* Sequences ending mid-word produce partial words
* Streaming generation requires handling incomplete subword tokens

'''Special Token Artifacts:'''
* Multiple consecutive special tokens (e.g., multiple EOS)
* Special tokens in unexpected positions

'''Byte-Level Issues:'''
* Invalid UTF-8 sequences from byte-level BPE
* Surrogate pairs and multi-byte characters

=== Design Considerations ===

* '''Performance''': Batch decoding for efficiency
* '''Streaming Support''': Incremental decoding for generation
* '''Error Handling''': Graceful degradation for invalid IDs
* '''Whitespace Preservation''': Balancing cleanup vs information retention
* '''Formatting Preservation''': Maintaining newlines, tabs, special formatting

=== Common Patterns ===

'''Generation Tasks:'''
<syntaxhighlight lang="python">
# Decode model output, skip special tokens
output_ids = model.generate(input_ids)
text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
</syntaxhighlight>

'''Streaming Decoding:'''
<syntaxhighlight lang="python">
# Decode incrementally as tokens are generated
generated_ids = []
for new_token_id in generate_stream():
    generated_ids.append(new_token_id)
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(text, end='', flush=True)
</syntaxhighlight>

'''Batch Processing:'''
<syntaxhighlight lang="python">
# Decode multiple sequences efficiently
batch_outputs = model.generate(batch_inputs, num_return_sequences=5)
texts = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)
</syntaxhighlight>

'''Debug Mode:'''
<syntaxhighlight lang="python">
# Keep special tokens to debug formatting issues
text_with_specials = tokenizer.decode(token_ids, skip_special_tokens=False)
print(text_with_specials)  # "[CLS] text [SEP]"
</syntaxhighlight>

=== Quality Considerations ===

* '''Subword Artifacts''': Visible word piece boundaries indicate decoding issues
* '''Extra Spaces''': Poor space cleanup degrades readability
* '''Repeated Tokens''': Generation failures manifest as repeated decoded text
* '''Truncation Effects''': Incomplete sentences from length limits
* '''Language Mixing''': Multilingual models may produce mixed-language output

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_transformers_tokenizer_decode]]
