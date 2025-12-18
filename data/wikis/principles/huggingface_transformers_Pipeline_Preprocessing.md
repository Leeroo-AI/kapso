{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|Transformers Documentation|https://huggingface.co/docs/transformers]]
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
|-
! Domains
| [[domain::NLP]], [[domain::Inference]]
|-
! Last Updated
| [[last_updated::2025-12-18 00:00 GMT]]
|}

== Overview ==
Pipeline preprocessing transforms raw, task-specific inputs into standardized tensor representations suitable for neural network forward passes.

=== Description ===
Pipeline preprocessing is the principle of converting raw user inputs (text strings, images, audio waveforms, structured data) into the specific tensor format required by a model's forward method. This transformation is task-dependent: text classification requires tokenized input IDs and attention masks, image classification needs normalized pixel tensors, question answering combines question and context encodings, and multimodal tasks merge multiple input modalities. Rather than requiring users to understand the intricacies of each model's input format, preprocessing encapsulates this complexity behind a consistent interface.

The preprocessing step handles tokenization (splitting text into subwords/tokens), encoding (converting tokens to IDs), special token insertion (CLS, SEP, padding tokens), attention mask generation (distinguishing real tokens from padding), image transformations (resizing, normalization, format conversion), audio feature extraction (spectrograms, mel-frequency coefficients), padding and truncation (ensuring consistent tensor shapes), and batching (combining multiple inputs). This separation of concerns allows preprocessing to run on CPU while model inference runs on GPU, enables efficient batching strategies, and provides clear boundaries for testing and optimization.

=== Usage ===
Apply this principle when:
* Designing inference pipelines that need consistent input handling across modalities
* Implementing custom pipeline classes for new task types
* Building APIs where users provide raw data and expect processed results
* Optimizing inference throughput through preprocessing/inference parallelization
* Creating reproducible model evaluation pipelines where preprocessing must match training

== Theoretical Basis ==

=== Preprocessing Interface Contract ===

All pipeline preprocessing methods follow a standard signature:

<pre>
FUNCTION preprocess(input: Any, **preprocess_parameters: Dict) -> Dict[str, Tensor]:
    """
    Args:
        input: Raw input in task-appropriate format
        preprocess_parameters: Task-specific preprocessing configuration

    Returns:
        Dictionary of tensors ready for model forward pass

    Contract Requirements:
        1. Output must be dict with string keys and tensor values
        2. Output keys must match model's forward signature expectations
        3. Tensors must be on CPU (device transfer happens in forward step)
        4. Batch dimension included even for single inputs
    """
</pre>

=== Task-Specific Input Formats ===

Different tasks accept different raw input types:

<pre>
Text Classification:
    input: str OR List[str]
    output: {
        "input_ids": Tensor[batch, seq_len],
        "attention_mask": Tensor[batch, seq_len],
        "token_type_ids": Tensor[batch, seq_len]  # Optional
    }

Image Classification:
    input: PIL.Image OR np.ndarray OR str (path) OR List[...]
    output: {
        "pixel_values": Tensor[batch, channels, height, width]
    }

Question Answering:
    input: {"question": str, "context": str}
    output: {
        "input_ids": Tensor[batch, seq_len],
        "attention_mask": Tensor[batch, seq_len],
        "token_type_ids": Tensor[batch, seq_len]  # Separates Q from context
    }

Token Classification (NER):
    input: str OR List[str]
    output: {
        "input_ids": Tensor[batch, seq_len],
        "attention_mask": Tensor[batch, seq_len],
        "offset_mapping": List[Tuple]  # For token→character alignment
    }

Image-to-Text (Captioning):
    input: PIL.Image OR np.ndarray OR str (path)
    output: {
        "pixel_values": Tensor[batch, channels, height, width],
        "decoder_input_ids": Tensor[batch, 1]  # Start token
    }

Multimodal (VQA):
    input: {"image": PIL.Image, "question": str}
    output: {
        "pixel_values": Tensor[batch, channels, height, width],
        "input_ids": Tensor[batch, seq_len],
        "attention_mask": Tensor[batch, seq_len]
    }
</pre>

=== Preprocessing Operations by Modality ===

'''Text Preprocessing:'''
<pre>
1. Tokenization
   text = "Hello world"
   tokens = ["[CLS]", "Hello", "world", "[SEP]"]

2. Encoding
   token_ids = [101, 7592, 2088, 102]

3. Attention Mask Generation
   # 1 for real tokens, 0 for padding
   attention_mask = [1, 1, 1, 1]

4. Padding/Truncation
   IF len(token_ids) < max_length:
       token_ids = token_ids + [PAD] * (max_length - len(token_ids))
       attention_mask = attention_mask + [0] * (max_length - len(attention_mask))
   ELSE IF len(token_ids) > max_length:
       token_ids = token_ids[:max_length]
       attention_mask = attention_mask[:max_length]

5. Tensor Conversion
   input_ids = torch.tensor([token_ids])
   attention_mask = torch.tensor([attention_mask])
</pre>

'''Image Preprocessing:'''
<pre>
1. Format Conversion
   IF input is PIL.Image:
       image = np.array(input)
   ELSE IF input is path string:
       image = np.array(PIL.Image.open(input))

2. Resizing
   image = resize(image, size=(224, 224))

3. Normalization
   # Per-channel normalization
   image = (image - mean) / std
   # Where mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] for ImageNet

4. Channel Reordering
   # PIL/numpy: (height, width, channels) → PyTorch: (channels, height, width)
   image = image.transpose(2, 0, 1)

5. Tensor Conversion
   pixel_values = torch.tensor(image).unsqueeze(0)  # Add batch dim
</pre>

'''Audio Preprocessing:'''
<pre>
1. Loading and Resampling
   waveform, sample_rate = load_audio(audio_path)
   IF sample_rate != target_sample_rate:
       waveform = resample(waveform, sample_rate, target_sample_rate)

2. Feature Extraction
   # Compute STFT and mel-spectrogram
   spectrogram = compute_mel_spectrogram(waveform)

3. Normalization
   spectrogram = (spectrogram - mean) / std

4. Padding/Truncation
   IF spectrogram.shape[1] < max_length:
       spectrogram = pad(spectrogram, max_length)
   ELSE:
       spectrogram = spectrogram[:, :max_length]

5. Tensor Conversion
   input_values = torch.tensor(spectrogram).unsqueeze(0)
</pre>

=== Parameter Sanitization ===

Preprocessing parameters flow through a sanitization step:

<pre>
FUNCTION _sanitize_parameters(**kwargs) -> (preprocess_params, forward_params, postprocess_params):
    """
    Separates user kwargs into preprocessing, forward, and postprocessing params.

    Example for text generation:
        kwargs = {
            "truncation": True,        → preprocess_params
            "max_length": 512,         → preprocess_params
            "do_sample": True,         → forward_params (generation)
            "temperature": 0.8,        → forward_params
            "clean_up_tokenization_spaces": True  → postprocess_params
        }
    """
    preprocess_params = {}
    forward_params = {}
    postprocess_params = {}

    # Task-specific logic to route parameters
    FOR key, value IN kwargs:
        IF key in PREPROCESS_PARAM_NAMES:
            preprocess_params[key] = value
        ELSE IF key in FORWARD_PARAM_NAMES:
            forward_params[key] = value
        ELSE IF key in POSTPROCESS_PARAM_NAMES:
            postprocess_params[key] = value

    RETURN preprocess_params, forward_params, postprocess_params
</pre>

=== Batch Processing ===

Preprocessing supports both single and batch inputs:

<pre>
Single Input Path:
    input = "Hello"
    preprocessed = preprocess(input)
    # preprocessed["input_ids"].shape = [1, seq_len]

Batch Input Path:
    inputs = ["Hello", "World", "This is a longer sentence"]
    preprocessed = preprocess(inputs, padding=True)
    # preprocessed["input_ids"].shape = [3, max_seq_len]
    # Automatic padding to longest in batch

Streaming Path (for large datasets):
    FOR input IN input_stream:
        preprocessed = preprocess(input)
        YIELD preprocessed
</pre>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_transformers_Pipeline_preprocess]]
