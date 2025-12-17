{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|Transformers Docs|https://huggingface.co/docs/transformers]]
|-
! Domains
| [[domain::NLP]], [[domain::Inference]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

== Overview ==

Pipeline preprocessing is the transformation of raw user inputs into model-compatible tensor representations.

=== Description ===

Pipeline preprocessing addresses the impedance mismatch between human-interpretable data formats (strings, images, audio files) and the numerical tensor representations required by neural networks. This transformation is modality-specific and model-specific: text requires tokenization (mapping words/subwords to integer indices), images require normalization and resizing (converting PIL Images to normalized float tensors), audio requires feature extraction (converting waveforms to spectrograms or mel-frequency coefficients), and multimodal tasks require coordinated processing of multiple input types.

The preprocessing principle establishes a contract: accept raw inputs in natural formats and produce dictionaries of tensors that satisfy the model's forward pass interface. This contract enables separation of concerns—the model need not know about string encoding or image formats, and preprocessing need not know about model architectures. The transformation must be deterministic and reversible where appropriate (e.g., token IDs should be decodable back to text).

Preprocessing also handles batching concerns: inputs may arrive as single items or sequences, and preprocessing must handle padding, truncation, and attention masking to create uniform-sized batches. The pattern supports parameterization, allowing users to control truncation strategies, maximum sequence lengths, padding sides, and other task-specific options through a consistent parameter passing mechanism.

=== Usage ===

Use pipeline preprocessing when you need to:
* Convert raw data formats to model-compatible tensors
* Apply modality-specific transformations (tokenization, normalization, feature extraction)
* Handle variable-length inputs with padding and truncation
* Create attention masks and other auxiliary tensors
* Support both single-item and batch processing
* Parameterize preprocessing behavior (max_length, truncation, padding)

== Theoretical Basis ==

Pipeline preprocessing follows a modality-specific transformation pipeline pattern:

```
Input: raw_input (str, PIL.Image, np.ndarray, etc.), preprocessing_params

For Text Modality:
  Step 1: Tokenization
    tokens = tokenizer.tokenize(raw_input)
    # Converts "Hello world" -> ["Hello", "world"]

  Step 2: Token-to-ID Mapping
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    # Converts ["Hello", "world"] -> [7592, 2088]

  Step 3: Special Token Addition
    input_ids = [CLS] + input_ids + [SEP]
    # Adds model-specific special tokens

  Step 4: Padding/Truncation
    if len(input_ids) > max_length:
      input_ids = input_ids[:max_length]
    elif len(input_ids) < max_length:
      input_ids = input_ids + [PAD] * (max_length - len(input_ids))

  Step 5: Attention Mask Creation
    attention_mask = [1 if token != PAD else 0 for token in input_ids]

  Step 6: Tensorization
    return {
      "input_ids": torch.tensor([input_ids]),
      "attention_mask": torch.tensor([attention_mask])
    }

For Image Modality:
  Step 1: Resizing
    image = resize(raw_input, size=(224, 224))

  Step 2: Normalization
    image = (image - mean) / std
    # Apply channel-wise normalization

  Step 3: Channel Reordering
    image = transpose(image, (2, 0, 1))
    # Convert HWC to CHW format

  Step 4: Tensorization
    return {
      "pixel_values": torch.tensor([image])
    }

For Audio Modality:
  Step 1: Resampling
    audio = resample(raw_input, target_sr=16000)

  Step 2: Feature Extraction
    features = extract_features(audio)
    # Could be mel-spectrogram, MFCC, etc.

  Step 3: Normalization
    features = (features - mean) / std

  Step 4: Tensorization
    return {
      "input_features": torch.tensor([features])
    }

For Multimodal Tasks:
  # Combine preprocessing from multiple modalities
  text_features = preprocess_text(text_input)
  image_features = preprocess_image(image_input)

  return {
    **text_features,
    **image_features
  }

Output: dict[str, torch.Tensor]
```

Key principles:

1. **Determinism**: Same input always produces same output
2. **Model Specificity**: Preprocessing matches model's training procedure
3. **Batch Dimension**: All tensors include batch dimension (even for single inputs)
4. **Auxiliary Tensors**: Include attention masks, token type IDs, position IDs as needed
5. **Parameter Transparency**: User parameters (max_length, padding) flow through cleanly
6. **Format Standardization**: Output dictionary keys follow consistent naming (input_ids, pixel_values, input_features)

The preprocessing stage is stateless and side-effect-free, making it safe for parallelization and caching. It establishes the data contract between user inputs and model expectations.

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_transformers_Pipeline_preprocess]]

=== Part Of ===
* [[part_of::Principle:huggingface_transformers_Pipeline_Instantiation]]

=== Feeds Into ===
* [[feeds_into::Principle:huggingface_transformers_Pipeline_Model_Forward]]
