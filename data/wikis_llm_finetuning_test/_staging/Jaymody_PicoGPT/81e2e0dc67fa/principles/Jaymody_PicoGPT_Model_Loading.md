# Principle: Model_Loading

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|Language Models are Unsupervised Multitask Learners|https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf]]
* [[source::Blog|OpenAI GPT-2|https://openai.com/blog/better-language-models/]]
* [[source::Repo|PicoGPT|https://github.com/jaymody/picoGPT]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::NLP]], [[domain::Model_Management]]
|-
! Last Updated
| [[last_updated::2026-01-14 10:00 GMT]]
|}

== Overview ==

Mechanism for loading pre-trained GPT-2 model weights from TensorFlow checkpoints into NumPy arrays for inference.

=== Description ===

Model Loading is the process of retrieving pre-trained neural network parameters from persistent storage and restructuring them for use in inference. In the context of GPT-2, this involves:

1. **Downloading checkpoint files** from OpenAI's public storage if not already cached locally
2. **Parsing TensorFlow checkpoint format** to extract weight tensors
3. **Restructuring weights** from flat TensorFlow naming convention into nested Python dictionaries that match the model architecture
4. **Loading the tokenizer** (encoder) that maps between text and token IDs

This principle addresses the practical challenge of using pre-trained models: weights are typically stored in framework-specific formats (TensorFlow, PyTorch) and must be converted to the inference framework's expected format. PicoGPT demonstrates this by loading TensorFlow checkpoints into pure NumPy arrays.

=== Usage ===

Use this principle when:
- Initializing a pre-trained model for inference or fine-tuning
- Converting model weights between frameworks (e.g., TensorFlow → NumPy)
- Implementing lazy/on-demand model downloading
- Building educational implementations that expose weight loading internals

This is typically the first step in any inference pipeline and must complete before tokenization or forward pass can occur.

== Theoretical Basis ==

GPT-2 weights are organized hierarchically:

```
model/
├── wte (token embeddings): [n_vocab, n_embd]
├── wpe (position embeddings): [n_ctx, n_embd]
├── h{i}/ (transformer blocks 0..n_layer-1)
│   ├── ln_1/ (layer norm 1)
│   │   ├── g (gamma): [n_embd]
│   │   └── b (beta): [n_embd]
│   ├── attn/
│   │   ├── c_attn/ (QKV projection)
│   │   │   ├── w: [n_embd, 3*n_embd]
│   │   │   └── b: [3*n_embd]
│   │   └── c_proj/ (output projection)
│   │       ├── w: [n_embd, n_embd]
│   │       └── b: [n_embd]
│   ├── ln_2/ (layer norm 2)
│   └── mlp/
│       ├── c_fc/ (up projection)
│       │   ├── w: [n_embd, 4*n_embd]
│       │   └── b: [4*n_embd]
│       └── c_proj/ (down projection)
│           ├── w: [4*n_embd, n_embd]
│           └── b: [n_embd]
└── ln_f/ (final layer norm)
    ├── g: [n_embd]
    └── b: [n_embd]
```

**Pseudo-code for checkpoint parsing:**

<syntaxhighlight lang="python">
# Abstract algorithm (not actual implementation)
params = {"blocks": [{}] * n_layer}
for variable_name, value in checkpoint:
    if variable_name starts with "h{i}/":
        # Extract layer index and sub-path
        layer_idx, sub_path = parse(variable_name)
        params["blocks"][layer_idx][sub_path] = value
    else:
        # Top-level params (wte, wpe, ln_f)
        params[variable_name] = value
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Jaymody_PicoGPT_Load_Encoder_Hparams_And_Params]]

=== Uses Heuristics ===
* [[uses_heuristic::Heuristic:Jaymody_PicoGPT_Model_Size_Memory_Requirements]]
