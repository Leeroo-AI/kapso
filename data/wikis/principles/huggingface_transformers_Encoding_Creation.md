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
Packaging tokenized outputs into a structured container with input IDs, attention masks, and metadata for model consumption.

=== Description ===
Encoding creation is the final step of the tokenization pipeline, wrapping the various outputs (input_ids, attention_mask, token_type_ids, special_tokens_mask) into a unified BatchEncoding object. This principle solves the problem of managing multiple related tensors and providing a convenient interface for accessing tokenization outputs. It fits at the end of the tokenization pipeline, after all processing is complete and before passing to the model.

The BatchEncoding object acts as a dictionary with special features: dict-like access to components, attribute-style access for convenience, tensor conversion capabilities, support for fast tokenizer features (word_ids, sequence_ids, offsets), batch indexing and slicing, and preservation of encoding metadata. This unified interface simplifies model input preparation and enables advanced features like extractive question answering.

=== Usage ===
This principle should be applied when:
* Returning tokenizer outputs from encode/encode_plus/__call__ methods
* Packaging multiple encoding components together
* Providing a convenient interface for model inputs
* Enabling tensor conversion (PyTorch, TensorFlow, NumPy)
* Supporting advanced features like token-to-word mapping

== Theoretical Basis ==
The encoding creation principle follows these logical steps:

1. '''Data Collection''': Gather all encoding components
   * '''input_ids''': Token IDs representing the encoded text
   * '''attention_mask''': Binary mask indicating real vs padding tokens
   * '''token_type_ids''': Segment IDs for sequence pairs (0 for seq1, 1 for seq2)
   * '''special_tokens_mask''': Binary mask indicating special tokens
   * '''offset_mapping''': Character offsets for each token (fast tokenizers only)
   * '''overflowing_tokens''': Tokens that exceeded max_length with stride

2. '''Dictionary Construction''': Build base data structure
   * Create dict with component names as keys
   * Store lists (for batches) or single values
   * Example: {"input_ids": [101, 7592, 102], "attention_mask": [1, 1, 1]}
   * Support both single sequences and batches

3. '''Encoding Attachment''': Include fast tokenizer encoding objects
   * For fast tokenizers, attach tokenizers.Encoding objects
   * Encoding objects provide advanced features:
     - tokens(): Get token strings
     - word_ids(): Map tokens to original words
     - sequence_ids(): Map tokens to sequence (0 or 1)
     - offsets: Character offsets in original text
   * None for slow tokenizers

4. '''Sequence Count Tracking''': Record number of input sequences
   * n_sequences = 1: Single sequence
   * n_sequences = 2: Sequence pair
   * n_sequences = None: Unknown
   * Used for validation and processing

5. '''Tensor Conversion''': Optionally convert to tensor format
   * '''PyTorch''': Convert lists to torch.Tensor
   * '''TensorFlow''': Convert lists to tf.Tensor
   * '''NumPy''': Convert lists to np.ndarray
   * Support prepend_batch_axis for single sequences
   * Preserve data types (integers for IDs, floats if needed)

6. '''Interface Implementation''': Provide access methods
   * '''Dict-style access''': encoding["input_ids"]
   * '''Attribute access''': encoding.input_ids
   * '''Batch indexing''': encoding[0] gets first sequence
   * '''Slicing''': encoding[:2] gets first two sequences
   * '''Key iteration''': for key in encoding.keys()

7. '''Method Delegation''': Provide tensor operations
   * to(): Move tensors to device (PyTorch)
   * numpy(): Convert to NumPy arrays
   * tolist(): Convert to Python lists
   * items(), keys(), values(): Dict-like iteration

Pseudocode:
```
class BatchEncoding:
    def __init__(self, data, encoding, tensor_type, prepend_batch_axis, n_sequences):
        self.data = data  # Dict with input_ids, attention_mask, etc.
        self._encodings = encoding  # Fast tokenizer Encoding objects
        self._n_sequences = n_sequences

        if tensor_type:
            self.convert_to_tensors(tensor_type, prepend_batch_axis)

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.data[key]  # Dict access: encoding["input_ids"]
        elif isinstance(key, int):
            return self._encodings[key]  # Batch indexing
        elif isinstance(key, slice):
            return {k: v[key] for k, v in self.data.items()}  # Slicing

    def __getattr__(self, key):
        return self.data[key]  # Attribute access: encoding.input_ids

    def convert_to_tensors(self, tensor_type, prepend_batch_axis):
        if tensor_type == "pt":
            import torch
            for key in self.data:
                self.data[key] = torch.tensor(self.data[key])
                if prepend_batch_axis:
                    self.data[key] = self.data[key].unsqueeze(0)
        elif tensor_type == "tf":
            import tensorflow as tf
            for key in self.data:
                self.data[key] = tf.constant(self.data[key])
        elif tensor_type == "np":
            import numpy as np
            for key in self.data:
                self.data[key] = np.array(self.data[key])

    def to(self, device):
        # Move PyTorch tensors to device
        for key in self.data:
            if hasattr(self.data[key], 'to'):
                self.data[key] = self.data[key].to(device)
        return self

function create_encoding(input_ids, attention_mask, token_type_ids, encoding, n_sequences):
    data = {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }

    if token_type_ids is not None:
        data["token_type_ids"] = token_type_ids

    return BatchEncoding(
        data=data,
        encoding=encoding,
        tensor_type=None,
        prepend_batch_axis=False,
        n_sequences=n_sequences
    )
```

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_transformers_BatchEncoding_creation]]
