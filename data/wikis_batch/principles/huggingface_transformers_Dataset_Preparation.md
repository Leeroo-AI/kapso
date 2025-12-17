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

Dataset preparation is the process of transforming raw data into the numerical representations required by neural network models.

=== Description ===

Dataset preparation addresses the fundamental incompatibility between human-readable data (text, images, audio) and neural network inputs (tensors of numerical values). For natural language processing, this involves tokenization—converting text into sequences of integer token IDs that correspond to entries in a model's vocabulary. The process must handle variable-length inputs, special tokens (start, end, padding, unknown), and create auxiliary tensors like attention masks that indicate which positions contain real data versus padding.

Efficient dataset preparation is critical for training performance. Rather than tokenizing examples one at a time during training, best practice is to pre-process the entire dataset once, converting all text to token IDs and creating a cached version. This prevents redundant computation and allows training loops to focus purely on model updates. Modern approaches use batched processing and multiprocessing to maximize throughput during preparation.

The preparation step also handles data cleaning, removing unnecessary columns, and ensuring consistent formatting across the dataset. It bridges the gap between data storage formats (CSV, JSON, databases) and the tensor-based inputs required by training loops.

=== Usage ===

Use dataset preparation when transitioning from raw data to model training. This is a mandatory first step for any supervised learning task—you cannot train on raw text or images directly. Apply dataset preparation when you have collected data in human-readable format and need to convert it to model inputs, when switching between different models that use different vocabularies or tokenization schemes, or when you need to create augmented versions of data (like adding special tokens or truncating sequences).

This step is particularly important when working with large datasets, as proper preparation with caching can save hours of redundant computation during iterative model development.

== Theoretical Basis ==

Dataset preparation transforms raw data D_raw into model-compatible tensors D_prepared through a series of deterministic transformations.

'''Tokenization Process:'''

For text input x = "Hello world":
1. Text normalization: lowercase, unicode normalization
2. Tokenization: Split into tokens ["hello", "world"]
3. Vocabulary lookup: Map to token IDs [2054, 2088]
4. Add special tokens: [CLS] x [SEP] → [101, 2054, 2088, 102]
5. Create attention mask: [1, 1, 1, 1] (all real tokens)

'''Mathematical Representation:'''

Given raw dataset D_raw = {(x_i, y_i)}_{i=1}^N where x_i is text and y_i is label:

tokenize(x_i) → {
    input_ids: [t_1, t_2, ..., t_L] where t_j ∈ [0, V-1],
    attention_mask: [a_1, a_2, ..., a_L] where a_j ∈ {0, 1},
    token_type_ids: [s_1, s_2, ..., s_L] where s_j ∈ {0, 1} (for dual sequences)
}

where:
* V = vocabulary size
* L = sequence length (padded/truncated to fixed length)
* a_j = 1 if position j contains real token, 0 if padding

'''Batched Processing:'''

Applying tokenization individually:
for each x_i in D_raw:
    process(x_i) → O(N) function calls

Applying tokenization in batches:
for each batch B in chunks(D_raw, batch_size=1000):
    process(B) → O(N/1000) function calls + vectorization benefits

'''Pseudocode:'''
<syntaxhighlight lang="text">
function prepare_dataset(raw_dataset, tokenizer, max_length):
    """
    Transform raw text dataset into tokenized tensors

    Args:
        raw_dataset: Collection of {text: str, label: int} examples
        tokenizer: Vocabulary and tokenization rules
        max_length: Maximum sequence length (truncate/pad to this)

    Returns:
        prepared_dataset: {input_ids, attention_mask, labels}
    """

    function tokenize_batch(examples):
        # Process multiple examples simultaneously
        texts = examples["text"]

        # Apply tokenization with vocabulary lookup
        tokenized = tokenizer(
            texts,
            truncation=True,      # Handle sequences > max_length
            padding="max_length", # Or pad dynamically later
            max_length=max_length,
            return_tensors=True
        )

        # tokenized contains:
        # - input_ids: [batch_size, max_length] integers
        # - attention_mask: [batch_size, max_length] binary
        # - optionally token_type_ids for sentence pairs

        return tokenized

    # Apply transformation to entire dataset efficiently
    prepared_dataset = raw_dataset.map(
        tokenize_batch,
        batched=True,           # Process in batches
        batch_size=1000,        # Vectorize 1000 examples at once
        num_proc=4,             # Use 4 CPU cores
        remove_columns=["text"] # Remove raw text to save memory
    )

    return prepared_dataset


# Example with specific vocabulary
tokenizer.vocab = {
    "[PAD]": 0,
    "[UNK]": 1,
    "[CLS]": 101,
    "[SEP]": 102,
    "hello": 2054,
    "world": 2088,
    ...
}

input_text = "Hello world"
# After preparation:
# input_ids:      [101, 2054, 2088, 102, 0, 0, ..., 0]  # padded to max_length
# attention_mask: [1,   1,    1,    1,   0, 0, ..., 0]  # 1s for real tokens
</syntaxhighlight>

'''Key Considerations:'''
* **Caching**: Prepared datasets should be cached to avoid redundant tokenization
* **Memory vs Compute**: Padding strategies trade memory (fixed-length) vs computation (dynamic padding)
* **Vocabulary Consistency**: The tokenizer vocabulary must match the model's vocabulary
* **Special Tokens**: Different models use different special tokens (BERT uses [CLS]/[SEP], GPT uses different scheme)
* **Truncation Strategy**: Long sequences can be truncated from left, right, or split into chunks

The efficiency of dataset preparation directly impacts iteration speed during model development, making batched and cached preprocessing essential for productive workflows.

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_transformers_Dataset_Tokenization]]
