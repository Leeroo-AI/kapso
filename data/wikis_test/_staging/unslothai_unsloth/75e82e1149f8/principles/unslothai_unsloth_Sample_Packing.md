# Principle: Sample Packing

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|Scaling Data-Constrained Language Models|https://arxiv.org/abs/2305.16264]]
* [[source::Doc|TRL Packing Documentation|https://huggingface.co/docs/trl/sft_trainer#packing]]
* [[source::Blog|Efficient Training Techniques|https://huggingface.co/docs/transformers/perf_train_gpu_one]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Training_Efficiency]], [[domain::NLP]]
|-
! Last Updated
| [[last_updated::2025-12-15 20:00 GMT]]
|}

== Overview ==
Training efficiency technique that concatenates multiple short sequences into single training examples up to the maximum sequence length, eliminating padding waste and improving GPU utilization.

=== Description ===
Sample packing addresses the inefficiency of padding in sequence-to-sequence training. When training examples have varying lengths, standard batching pads shorter sequences to match the longest, wasting computation on padding tokens.

'''The Padding Problem:'''
- Short examples (100 tokens) padded to max_length (2048) = 95% waste
- Batch of varying lengths requires padding to longest
- GPU compute wasted on attention over padding tokens

'''Packing Solution:'''
- Concatenate multiple examples into one sequence
- Separate with EOS tokens to maintain boundaries
- Use attention masks to prevent cross-example attention
- Result: Near 100% token utilization

'''Benefits:'''
- 2-5x effective batch size increase
- Better GPU utilization
- Faster training per epoch
- No quality degradation with proper masking

=== Usage ===
Enable sample packing when:
- Training on short-example datasets (< 512 tokens average)
- Dataset has high length variance
- GPU utilization is low during training
- Training time is a priority

'''Configuration:'''
- `packing=True` in SFTConfig
- Unsloth auto-enables padding-free mode with packing
- Works with both regular and chat-formatted data

== Theoretical Basis ==
'''Standard Padding:'''

<syntaxhighlight lang="python">
# Without packing: significant waste
def standard_batching(examples, max_length=2048):
    """Pad all sequences to max_length."""
    batch = []
    for ex in examples:
        # Pad short sequences
        padded = ex + [PAD_TOKEN] * (max_length - len(ex))
        batch.append(padded)

    # If examples are [100, 150, 200] tokens:
    # Total computation: 3 * 2048 = 6144 tokens
    # Actual content: 450 tokens
    # Efficiency: 450 / 6144 = 7.3%

    return batch
</syntaxhighlight>

'''Sample Packing:'''
<syntaxhighlight lang="python">
def pack_sequences(examples, max_length=2048, eos_token=2):
    """Pack multiple sequences into single training examples."""
    packed = []
    current_pack = []
    current_length = 0

    for ex in examples:
        ex_with_eos = ex + [eos_token]  # Add separator
        ex_len = len(ex_with_eos)

        if current_length + ex_len <= max_length:
            # Fits in current pack
            current_pack.extend(ex_with_eos)
            current_length += ex_len
        else:
            # Start new pack
            if current_pack:
                # Pad remaining space (minimal)
                current_pack += [PAD_TOKEN] * (max_length - current_length)
                packed.append(current_pack)
            current_pack = ex_with_eos
            current_length = ex_len

    # Handle last pack
    if current_pack:
        current_pack += [PAD_TOKEN] * (max_length - current_length)
        packed.append(current_pack)

    return packed

# If examples are [100, 150, 200, 180, 120] tokens:
# Pack 1: [100] + [150] + [200] = 450 tokens (+ EOS tokens)
# Pack 2: [180] + [120] = 300 tokens
# Total computation: 2 * 2048 = 4096 tokens
# Actual content: 750 tokens
# Efficiency: 750 / 4096 = 18.3% (still includes some padding)
# vs Standard: 750 / (5 * 2048) = 7.3%
</syntaxhighlight>

'''Attention Masking for Packed Sequences:'''
<syntaxhighlight lang="python">
def create_packing_attention_mask(packed_sequence, eos_positions):
    """Create attention mask that prevents cross-example attention."""
    seq_len = len(packed_sequence)
    mask = torch.zeros(seq_len, seq_len)

    # Identify example boundaries
    boundaries = [0] + eos_positions + [seq_len]

    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        # Each example can only attend to itself
        mask[start:end, start:end] = 1

    return mask

# Example:
# Packed: [Ex1...EOS][Ex2...EOS][Ex3...PAD]
# Attention mask:
# [1 1 1 0 0 0 0 0 0 0]  <- Ex1 attends to Ex1
# [1 1 1 0 0 0 0 0 0 0]
# [1 1 1 0 0 0 0 0 0 0]
# [0 0 0 1 1 1 0 0 0 0]  <- Ex2 attends to Ex2
# [0 0 0 1 1 1 0 0 0 0]
# [0 0 0 1 1 1 0 0 0 0]
# [0 0 0 0 0 0 1 1 0 0]  <- Ex3 attends to Ex3
# [0 0 0 0 0 0 1 1 0 0]
</syntaxhighlight>

'''Label Masking:'''
<syntaxhighlight lang="python">
def create_packing_labels(packed_sequence, boundaries, input_portions):
    """Create labels that train only on response portions."""
    labels = packed_sequence.clone()

    for i, (start, end) in enumerate(boundaries):
        # Mask input portion (user/system) with -100
        input_len = input_portions[i]
        labels[start:start + input_len] = -100

    # Mask all padding
    labels[labels == PAD_TOKEN] = -100

    return labels
</syntaxhighlight>

'''Efficiency Analysis:'''
<math>
Efficiency = \frac{\sum_{i} len(example_i)}{\lceil \frac{\sum_{i} len(example_i)}{max\_length} \rceil \times max\_length}
</math>

Approaches 100% as average example length approaches max_length or as dataset size increases.

== Related Pages ==
=== Implemented By ===
* [[implemented_by::Implementation:unslothai_unsloth_UnslothTrainer]]

=== Tips and Tricks ===
