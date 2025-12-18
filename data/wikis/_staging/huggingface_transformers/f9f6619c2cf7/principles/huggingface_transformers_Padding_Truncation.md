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
Standardizing sequence lengths in batches by adding padding tokens or removing excess tokens to meet length constraints.

=== Description ===
Padding and truncation are complementary operations that ensure all sequences in a batch have the same length, which is required for efficient tensor processing in neural networks. This principle solves the problem of variable-length text inputs: some sequences are too short (need padding) while others are too long (need truncation). It fits after encoding and before converting to tensors, transforming ragged sequences into uniform rectangular batches.

Padding adds special padding tokens (typically PAD token with ID 0) to sequences shorter than the target length, creating attention masks to indicate real vs padded positions. Truncation removes tokens from sequences longer than the maximum length, choosing which end to truncate and how to handle sequence pairs. These operations are essential for batch processing and respecting model maximum sequence length constraints.

=== Usage ===
This principle should be applied when:
* Creating batches of sequences with different lengths
* Preparing inputs for batch inference or training
* Respecting model maximum sequence length constraints (e.g., 512 for BERT)
* Enabling efficient tensor operations on GPUs/TPUs
* Creating data collators for PyTorch/TensorFlow data loaders

== Theoretical Basis ==
The padding and truncation principle follows these logical steps:

1. '''Length Analysis''': Determine current and target lengths
   * Measure current sequence length (number of token IDs)
   * Determine target length:
     - PaddingStrategy.LONGEST: max length in batch
     - PaddingStrategy.MAX_LENGTH: fixed max_length parameter
     - PaddingStrategy.DO_NOT_PAD: no padding applied
   * Identify which sequences need padding vs truncation

2. '''Truncation Strategy''': Reduce overly long sequences
   * '''Truncation options for single sequences''':
     - Truncate from right (default): Keep start, remove end
     - Truncate from left: Remove start, keep end
   * '''Truncation options for sequence pairs''':
     - LONGEST_FIRST: Alternately remove from longer sequence
     - ONLY_FIRST: Truncate only first sequence
     - ONLY_SECOND: Truncate only second sequence
   * Preserve special tokens during truncation
   * Maintain proper sequence pair structure

3. '''Padding Application''': Extend short sequences
   * '''Padding side''':
     - Right padding (default for BERT): [tokens] [PAD] [PAD]
     - Left padding (for GPT): [PAD] [PAD] [tokens]
   * Add padding token IDs (pad_token_id)
   * Calculate padding length: target_length - current_length
   * Apply padding to input_ids

4. '''Attention Mask Generation''': Mark real vs padding tokens
   * Real tokens: mask value = 1 (attend to these)
   * Padding tokens: mask value = 0 (ignore these)
   * Example: [1, 1, 1, 1, 0, 0] for 4 real tokens + 2 padding
   * Used by transformer attention mechanism to ignore padding

5. '''Token Type IDs Padding''': Extend segment IDs
   * Pad token_type_ids with pad_token_type_id (typically 0)
   * Maintain segment boundaries for sequence pairs
   * Example: [0, 0, 0, 1, 1, 0, 0] with padding at end

6. '''Special Token Mask Padding''': Extend special token indicators
   * Real tokens: 0
   * Special tokens: 1
   * Padding tokens: 1 (treated as special)
   * Used for masking during training

7. '''Pad to Multiple''': Align to hardware-efficient sizes
   * Pad to multiple of pad_to_multiple_of (e.g., 8 for Tensor Cores)
   * Improves GPU/TPU performance
   * Round up target length to next multiple
   * Example: length 67 → 72 when pad_to_multiple_of=8

Pseudocode:
```
function pad_sequence(input_ids, target_length, padding_side, pad_token_id):
    current_length = len(input_ids)

    if current_length >= target_length:
        return input_ids  # No padding needed

    padding_length = target_length - current_length

    if padding_side == "right":
        padded_ids = input_ids + [pad_token_id] * padding_length
    else:  # left padding
        padded_ids = [pad_token_id] * padding_length + input_ids

    return padded_ids

function truncate_sequence(input_ids, max_length, truncation_side):
    if len(input_ids) <= max_length:
        return input_ids  # No truncation needed

    if truncation_side == "right":
        return input_ids[:max_length]
    else:  # left truncation
        return input_ids[-max_length:]

function truncate_sequence_pair(ids1, ids2, max_length, strategy):
    total_length = len(ids1) + len(ids2)

    while total_length > max_length:
        if strategy == "LONGEST_FIRST":
            if len(ids1) > len(ids2):
                ids1 = ids1[:-1]
            else:
                ids2 = ids2[:-1]
        elif strategy == "ONLY_FIRST":
            ids1 = ids1[:-1]
        elif strategy == "ONLY_SECOND":
            ids2 = ids2[:-1]

        total_length = len(ids1) + len(ids2)

    return ids1, ids2

function create_attention_mask(input_ids, pad_token_id):
    return [1 if token_id != pad_token_id else 0 for token_id in input_ids]

function pad_batch(batch_input_ids, padding_strategy, max_length, pad_token_id):
    if padding_strategy == "LONGEST":
        target_length = max(len(ids) for ids in batch_input_ids)
    elif padding_strategy == "MAX_LENGTH":
        target_length = max_length
    else:
        return batch_input_ids  # No padding

    padded_batch = []
    attention_masks = []

    for input_ids in batch_input_ids:
        padded_ids = pad_sequence(input_ids, target_length, "right", pad_token_id)
        attention_mask = create_attention_mask(padded_ids, pad_token_id)

        padded_batch.append(padded_ids)
        attention_masks.append(attention_mask)

    return padded_batch, attention_masks
```

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_transformers_Batch_padding]]
