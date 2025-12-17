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

Managing variable-length sequences to meet fixed-size input requirements for neural network batching.

=== Description ===

Padding adds special tokens to short sequences to reach a target length, while truncation removes tokens from long sequences to fit within maximum length constraints. Both operations are essential for batching multiple sequences together, as neural networks require fixed-dimensional inputs. Padding uses designated PAD tokens that are masked in attention computations, and truncation can be applied from left or right side with strategies for sequence pairs. These operations balance model input requirements with information preservation.

=== Usage ===

Use padding when batching sequences of different lengths to ensure uniform tensor dimensions. Use truncation when sequences exceed model maximum length limits. Essential for efficient GPU/TPU processing, preventing out-of-memory errors, and maintaining consistent batch shapes throughout training and inference.

== Theoretical Basis ==

=== Core Concepts ===

'''Padding Strategies:'''
* '''No Padding''': Keep sequences as-is (only valid for batch size 1)
* '''Longest''': Pad all sequences to match longest in batch
* '''Max Length''': Pad all sequences to specified maximum length
* '''Multiple-of-N''': Pad to nearest multiple of N (for hardware optimization)

'''Truncation Strategies:'''
* '''Longest First''': Truncate longest sequence in pair until both fit
* '''Only First''': Truncate only first sequence in pair
* '''Only Second''': Truncate only second sequence in pair
* '''Do Not Truncate''': Raise error if sequences too long

'''Padding Side:'''
* '''Right Padding''': Add padding tokens at end (common for encoders)
* '''Left Padding''': Add padding tokens at start (common for generation)

=== Algorithm ===

<syntaxhighlight lang="text">
function PAD_SEQUENCES(sequences, padding_strategy, max_length, pad_token_id):
    // Determine target length
    if padding_strategy == "longest":
        target_length = max(len(seq) for seq in sequences)
    else if padding_strategy == "max_length":
        target_length = max_length
    else:
        return sequences  // No padding

    padded_sequences = []
    attention_masks = []

    for seq in sequences:
        pad_length = target_length - len(seq)

        if padding_side == "right":
            // Pad at end
            padded_seq = seq + [pad_token_id] * pad_length
            attention_mask = [1] * len(seq) + [0] * pad_length
        else:
            // Pad at start
            padded_seq = [pad_token_id] * pad_length + seq
            attention_mask = [0] * pad_length + [1] * len(seq)

        padded_sequences.append(padded_seq)
        attention_masks.append(attention_mask)

    return padded_sequences, attention_masks

function TRUNCATE_SEQUENCE(sequence, max_length, truncation_side):
    if len(sequence) <= max_length:
        return sequence

    if truncation_side == "right":
        return sequence[:max_length]
    else:  // left
        return sequence[-max_length:]

function TRUNCATE_SEQUENCE_PAIR(seq_a, seq_b, max_length, strategy):
    total_length = len(seq_a) + len(seq_b)

    if total_length <= max_length:
        return seq_a, seq_b

    if strategy == "longest_first":
        // Iteratively truncate longest until both fit
        while len(seq_a) + len(seq_b) > max_length:
            if len(seq_a) > len(seq_b):
                seq_a = seq_a[:-1]
            else:
                seq_b = seq_b[:-1]

    else if strategy == "only_first":
        overflow = total_length - max_length
        seq_a = seq_a[:len(seq_a) - overflow]

    else if strategy == "only_second":
        overflow = total_length - max_length
        seq_b = seq_b[:len(seq_b) - overflow]

    return seq_a, seq_b

function PAD_TO_MULTIPLE_OF(sequence, multiple, pad_token_id):
    current_length = len(sequence)
    remainder = current_length % multiple

    if remainder == 0:
        return sequence

    pad_length = multiple - remainder
    return sequence + [pad_token_id] * pad_length
</syntaxhighlight>

=== Mathematical Formulation ===

'''Padding Operation:'''

Given sequence <math>S = (s_1, s_2, \ldots, s_n)</math> and target length <math>L > n</math>:

* Right padding: <math>S' = (s_1, \ldots, s_n, p, \ldots, p)</math> where <math>|S'| = L</math>
* Left padding: <math>S' = (p, \ldots, p, s_1, \ldots, s_n)</math> where <math>|S'| = L</math>
* Attention mask: <math>M = (1^n, 0^{L-n})</math> for right, <math>M = (0^{L-n}, 1^n)</math> for left

'''Truncation Operation:'''

Given sequence <math>S = (s_1, s_2, \ldots, s_n)</math> and max length <math>L < n</math>:

* Right truncation: <math>S' = (s_1, s_2, \ldots, s_L)</math>
* Left truncation: <math>S' = (s_{n-L+1}, \ldots, s_n)</math>

'''Sequence Pair Truncation:'''

Given <math>S_A = (a_1, \ldots, a_m)</math> and <math>S_B = (b_1, \ldots, b_n)</math> with <math>m + n > L</math>:

* Longest first: Remove from longer sequence until <math>|S_A| + |S_B| \leq L</math>
* Preserve ratio: <math>|S_A'| : |S_B'| \approx |S_A| : |S_B|</math>

=== Key Properties ===

* '''Information Preservation''': Padding adds no information; truncation loses information
* '''Attention Masking''': Padded positions are masked (weight = 0) in self-attention
* '''Batch Uniformity''': All sequences in batch have same shape after padding
* '''Model Invariance''': Model output should not depend on padding (when masked properly)
* '''Efficiency Trade-off''': More padding = more wasted computation

=== Hardware Optimization ===

'''Tensor Core Utilization:'''

Modern GPUs (NVIDIA Volta+) achieve peak performance with specific tensor dimensions:
* Pad to multiples of 8 (FP16/BF16) or 16 (INT8) for optimal throughput
* Example: Sequence of length 67 → pad to 72 (next multiple of 8)

'''Memory Alignment:'''

CPU/GPU memory systems benefit from aligned memory access:
* Padding to powers of 2 can improve memory bandwidth
* Trade-off: More padding = more memory usage

=== Design Considerations ===

* '''Dynamic vs Static Padding''': Pad to longest in batch vs fixed maximum
* '''Padding Side Selection''': Right for encoding tasks, left for generation
* '''Truncation Strategy''': Balance between losing context and fitting in memory
* '''Stride for Truncation''': Keep overlapping windows when documents exceed max length
* '''Special Token Handling''': Ensure BOS/EOS tokens preserved after truncation

=== Common Patterns ===

'''Training:'''
* Dynamic padding to longest in batch (efficiency)
* Bucketing: Group similar-length sequences (reduce padding)
* Right padding for BERT-style models

'''Inference:'''
* Static padding to fixed length (consistency)
* Left padding for autoregressive generation (GPT-style)
* Batch size 1 often avoids padding

'''Question Answering:'''
* Truncate context but preserve full question
* Use stride for long documents (sliding window)
* Keep special tokens for segment separation

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_transformers_pad_truncate]]
