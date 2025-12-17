{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|Transformers Docs|https://huggingface.co/docs/transformers]]
|-
! Domains
| [[domain::NLP]], [[domain::Training]], [[domain::Data Processing]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

== Overview ==

Data collation is the process of combining individual dataset examples into uniform batches suitable for parallel processing on GPUs.

=== Description ===

Data collation solves a fundamental challenge in deep learning: neural networks process data in batches for efficiency, but individual examples often have different shapes (e.g., text sequences of varying lengths). GPUs require uniform tensor shapes within each batch to perform parallel matrix operations. Data collation transforms a list of variable-shaped examples into a single batch tensor by padding sequences to a common length, stacking tensors, and creating auxiliary tensors that track which positions contain real data versus padding.

The collation process occurs during training at batch creation time, just before data is fed to the model. It acts as a bridge between the dataset (which stores individual examples) and the model (which expects uniformly-shaped batches). Efficient collation strategies balance memory usage and computational overhead—padding all sequences to the maximum dataset length wastes memory, while padding to the longest sequence in each batch (dynamic padding) is memory-efficient but requires recalculation per batch.

Collation also handles label formatting, converting various label representations into the standard format expected by model loss functions, and manages data type conversions from Python types to framework-specific tensors (PyTorch, TensorFlow, JAX).

=== Usage ===

Use data collation whenever you're training models with variable-length inputs, which is nearly universal in NLP and common in other domains. Apply data collation when your dataset contains sequences of different lengths, when you want to optimize memory usage during training (dynamic padding is more efficient than static padding), or when you need to convert dataset examples into batch tensors on-the-fly. Data collation is essential for mini-batch gradient descent, which is the standard training paradigm in modern deep learning.

This pattern is particularly important for resource-constrained training where memory efficiency matters, or when working with datasets that have highly variable sequence lengths.

== Theoretical Basis ==

Data collation transforms a list of individual examples into a uniform batch tensor through padding and stacking operations.

'''Problem Statement:'''

Given a batch of N examples with variable lengths:
* Example 1: sequence of length L_1
* Example 2: sequence of length L_2
* ...
* Example N: sequence of length L_N

where L_1, L_2, ..., L_N may all be different.

Goal: Create uniform tensor of shape [N, L_max] where L_max is determined by padding strategy.

'''Padding Strategies:'''

1. **Dynamic Padding (Longest in Batch):**
   L_max = max(L_1, L_2, ..., L_N)
   - Memory efficient: only pad to longest in current batch
   - Different batches have different shapes
   - Preferred for most training scenarios

2. **Static Padding (Fixed Maximum):**
   L_max = MAX_LENGTH (constant)
   - All batches have identical shape
   - Wastes memory if sequences are typically short
   - Required for some hardware optimizations

3. **Padding to Multiple:**
   L_max = ceil(max(L_1, ..., L_N) / k) × k
   - Pad to nearest multiple of k (e.g., 8 for Tensor Cores)
   - Balances efficiency with hardware optimization

'''Collation Algorithm:'''
<syntaxhighlight lang="text">
function collate_batch(examples, pad_token_id=0):
    """
    Combine variable-length examples into uniform batch

    Args:
        examples: List of N dictionaries with keys:
            - input_ids: sequence of integers (length L_i)
            - attention_mask: sequence of 1s (length L_i)
            - labels: integer or sequence
        pad_token_id: Value to use for padding positions

    Returns:
        batch: Dictionary with tensors:
            - input_ids: [N, L_max]
            - attention_mask: [N, L_max]
            - labels: [N] or [N, L_max]
    """

    # Determine maximum length in batch
    L_max = max(len(ex["input_ids"]) for ex in examples)

    # Initialize batch tensors with padding value
    batch = {
        "input_ids": zeros([N, L_max]) + pad_token_id,
        "attention_mask": zeros([N, L_max]),
        "labels": zeros([N])  # or appropriate shape for labels
    }

    # Fill in actual values
    for i, example in enumerate(examples):
        L_i = len(example["input_ids"])

        # Copy sequence into padded tensor
        batch["input_ids"][i, :L_i] = example["input_ids"]

        # Attention mask: 1 for real tokens, 0 for padding
        batch["attention_mask"][i, :L_i] = 1

        # Labels
        batch["labels"][i] = example["label"]

    return batch


# Mathematical representation:
# For sequence i with length L_i < L_max:
#   input_ids[i] = [t_1, t_2, ..., t_{L_i}, pad, pad, ..., pad]
#                   |<---- L_i real ----->|<--(L_max - L_i)-->|
#
#   attention_mask[i] = [1, 1, ..., 1, 0, 0, ..., 0]
#                        |<-L_i->|<-(L_max-L_i)->|
</syntaxhighlight>

'''Example:'''
<syntaxhighlight lang="text">
Input examples:
Example 1: input_ids = [101, 2054, 2003, 102]              # length 4
Example 2: input_ids = [101, 2023, 2003, 1037, 3231, 102]  # length 6
Example 3: input_ids = [101, 2054, 102]                     # length 3

L_max = 6 (longest sequence in batch)

Output batch:
input_ids = [
    [101, 2054, 2003, 102,    0,    0],  # Example 1 + 2 padding
    [101, 2023, 2003, 1037, 3231, 102],  # Example 2 (no padding)
    [101, 2054, 102,    0,    0,    0],  # Example 3 + 3 padding
]

attention_mask = [
    [1, 1, 1, 1, 0, 0],  # 4 real tokens, 2 padding
    [1, 1, 1, 1, 1, 1],  # 6 real tokens, 0 padding
    [1, 1, 1, 0, 0, 0],  # 3 real tokens, 3 padding
]
</syntaxhighlight>

'''Attention Mask Usage in Model:'''

During self-attention computation, the attention mask prevents the model from attending to padding positions:

Attention_scores = Q @ K^T / sqrt(d_k)
Masked_scores = Attention_scores + (1 - attention_mask) × (-∞)
Attention_weights = softmax(Masked_scores)

Where (1 - attention_mask) converts:
* 1s (real tokens) → 0s (no change to score)
* 0s (padding) → 1s → multiply by -∞ → softmax → 0 attention weight

This ensures padding tokens don't influence model computations while allowing efficient batched processing.

'''Memory Efficiency:'''

Static padding: memory = N × MAX_LENGTH × d
Dynamic padding: memory = N × avg(L_max_per_batch) × d

For dataset with mostly short sequences but occasional long ones, dynamic padding can save 50%+ memory.

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_transformers_DataCollatorWithPadding]]
