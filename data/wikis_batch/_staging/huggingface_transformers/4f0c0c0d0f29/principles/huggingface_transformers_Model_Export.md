{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|Transformers Docs|https://huggingface.co/docs/transformers]]
|-
! Domains
| [[domain::NLP]], [[domain::Training]], [[domain::Model Serialization]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

== Overview ==

Model export is the process of serializing trained neural network parameters and configurations to disk for later reuse, deployment, or sharing.

=== Description ===

Model export addresses the need to persist trained models beyond the lifetime of a training process. After investing computational resources to train a model, the learned parameters must be saved to disk in a format that allows the model to be reloaded and used for inference, continued training, or transfer learning. Export involves serializing model weights (the numerical values of all parameters), model configuration (architecture specifications like number of layers, hidden dimensions, activation functions), and associated components (tokenizers, feature extractors, preprocessing configurations).

The export process must handle various distributed training scenarios where model parameters may be sharded across multiple devices, stored in different precision formats (FP32, FP16, quantized), or wrapped in distributed training containers. It must consolidate these distributed states into a single coherent checkpoint that can be loaded on different hardware configurations. The saved artifacts should be framework-compatible (loadable via standard APIs), human-readable (configuration in JSON), and optionally include metadata for reproducibility and model cards for documentation.

Model export transforms ephemeral training state into persistent, reusable assets that enable the full lifecycle of ML models: development → training → validation → deployment → monitoring → retraining. It's the bridge between training and production, between researchers and end users, and between different stages of model development.

=== Usage ===

Use model export after training completes to preserve the final model, periodically during training to create checkpoints, when achieving a performance milestone worth preserving, or when preparing models for deployment to production systems. Export models when you want to share them with others (via model hubs), when switching between different hardware for training vs inference, when creating model versioning for A/B testing, or when archiving models for reproducibility and compliance. Model export is essential for any model that will be used beyond the immediate training session.

== Theoretical Basis ==

Model export serializes the learned function f_θ by persisting its parameters θ and configuration C.

'''Components to Serialize:'''

1. **Model Parameters (θ):**
   - Weight matrices: W^(l) ∈ R^(d_{l+1} × d_l) for each layer l
   - Bias vectors: b^(l) ∈ R^(d_{l+1})
   - Normalization parameters: γ, β for LayerNorm/BatchNorm
   - Embedding matrices: E ∈ R^(V × d) for vocabulary V

2. **Model Configuration (C):**
   - Architecture specification: layer types, dimensions, activation functions
   - Model type: BERT, GPT, T5, etc.
   - Training configuration: task type, number of labels

3. **Associated Components:**
   - Tokenizer vocabulary and rules
   - Feature extractor configurations
   - Preprocessing parameters

'''Serialization Format:'''

<syntaxhighlight lang="text">
SavedModel = {
    "model_state": {
        "layer_0.weight": Tensor([...]),  # Shape: [hidden_dim, input_dim]
        "layer_0.bias": Tensor([...]),     # Shape: [hidden_dim]
        "layer_1.weight": Tensor([...]),
        ...
        "layer_n.weight": Tensor([...])
    },

    "config": {
        "model_type": "bert",
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 512,
        "vocab_size": 30522,
        "num_labels": 2  # For classification
    },

    "tokenizer": {
        "vocab": {"[PAD]": 0, "[UNK]": 1, ...},
        "do_lower_case": true,
        "model_max_length": 512
    },

    "metadata": {
        "framework": "pytorch",
        "transformers_version": "4.36.0",
        "torch_dtype": "float32"
    }
}
</syntaxhighlight>

'''Export Algorithm:'''

<syntaxhighlight lang="text">
function export_model(
    model: NeuralNetwork,
    output_dir: str,
    tokenizer: Tokenizer = None,
    save_config: bool = True,
    save_tokenizer: bool = True
) -> None:
    """
    Serialize trained model to disk

    Creates directory structure:
    output_dir/
        ├── pytorch_model.bin or model.safetensors  # Model weights
        ├── config.json                              # Model architecture config
        ├── tokenizer_config.json                    # Tokenizer settings
        ├── vocab.txt or vocab.json                  # Vocabulary
        ├── special_tokens_map.json                  # Special token mapping
        └── training_args.bin                        # Training configuration
    """

    # Create output directory
    create_directory(output_dir, exist_ok=True)

    # 1. Handle distributed training scenarios
    if is_distributed_training():
        # Consolidate parameters from multiple devices
        if is_zero_optimization():  # DeepSpeed ZeRO
            state_dict = consolidate_zero_checkpoints()
        elif is_fsdp():  # Fully Sharded Data Parallel
            state_dict = gather_full_state_dict()
        elif is_model_parallel():
            state_dict = gather_from_tensor_parallel()
        else:
            state_dict = model.state_dict()
    else:
        # Single device: directly get state dict
        state_dict = model.state_dict()

    # 2. Remove optimizer-specific keys if present
    state_dict = {
        k: v for k, v in state_dict.items()
        if not k.startswith("_optimizer") and not k.startswith("_scheduler")
    }

    # 3. Handle precision conversion if needed
    if save_in_fp16 and state_dict_is_fp32(state_dict):
        state_dict = convert_to_fp16(state_dict)

    # 4. Save model weights
    weights_path = os.path.join(output_dir, "pytorch_model.bin")

    # Option A: PyTorch native format
    torch.save(state_dict, weights_path)

    # Option B: SafeTensors format (safer, faster)
    # from safetensors.torch import save_file
    # save_file(state_dict, os.path.join(output_dir, "model.safetensors"))

    # 5. Save model configuration
    if save_config and hasattr(model, "config"):
        config_path = os.path.join(output_dir, "config.json")
        model.config.to_json_file(config_path)

    # 6. Save tokenizer if provided
    if save_tokenizer and tokenizer is not None:
        tokenizer.save_pretrained(output_dir)
        # Creates: tokenizer_config.json, vocab.txt/json, special_tokens_map.json

    # 7. Save training arguments for reproducibility
    if hasattr(self, "args"):
        args_path = os.path.join(output_dir, "training_args.bin")
        torch.save(self.args, args_path)

    print(f"Model saved to {output_dir}")


function load_model(model_path: str) -> NeuralNetwork:
    """
    Deserialize model from disk

    The inverse operation of export_model
    """

    # 1. Load configuration
    config_path = os.path.join(model_path, "config.json")
    config = ModelConfig.from_json_file(config_path)

    # 2. Initialize model architecture
    model = ModelClass.from_config(config)

    # 3. Load weights
    weights_path = os.path.join(model_path, "pytorch_model.bin")

    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location="cpu")
    else:
        # Try safetensors format
        from safetensors.torch import load_file
        state_dict = load_file(
            os.path.join(model_path, "model.safetensors")
        )

    # 4. Load weights into model
    model.load_state_dict(state_dict, strict=True)

    # 5. Load tokenizer if present
    tokenizer = None
    if os.path.exists(os.path.join(model_path, "tokenizer_config.json")):
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    return model, tokenizer
</syntaxhighlight>

'''State Dictionary Structure:'''

For a simple 2-layer transformer:
```
state_dict = {
    # Embedding layer
    "embeddings.word_embeddings.weight": [vocab_size, hidden_dim],
    "embeddings.position_embeddings.weight": [max_seq_len, hidden_dim],

    # Transformer layers
    "encoder.layer.0.attention.self.query.weight": [hidden_dim, hidden_dim],
    "encoder.layer.0.attention.self.query.bias": [hidden_dim],
    "encoder.layer.0.attention.self.key.weight": [hidden_dim, hidden_dim],
    "encoder.layer.0.attention.self.key.bias": [hidden_dim],
    "encoder.layer.0.attention.self.value.weight": [hidden_dim, hidden_dim],
    "encoder.layer.0.attention.self.value.bias": [hidden_dim],
    "encoder.layer.0.attention.output.dense.weight": [hidden_dim, hidden_dim],
    "encoder.layer.0.attention.output.dense.bias": [hidden_dim],
    "encoder.layer.0.intermediate.dense.weight": [intermediate_dim, hidden_dim],
    "encoder.layer.0.intermediate.dense.bias": [intermediate_dim],
    "encoder.layer.0.output.dense.weight": [hidden_dim, intermediate_dim],
    "encoder.layer.0.output.dense.bias": [hidden_dim],

    # Layer 1 (similar structure)
    ...

    # Classification head
    "classifier.weight": [num_labels, hidden_dim],
    "classifier.bias": [num_labels]
}
```

Total parameters = Σ (size of each tensor)
For BERT-base: ~110M parameters ≈ 440MB in FP32, ≈ 220MB in FP16

'''Distributed Training Considerations:'''

**DeepSpeed ZeRO Stage 3:**
- Parameters sharded across devices
- Must gather full state dict before saving
- Only main process should save

**FSDP (Fully Sharded Data Parallel):**
- Each device holds 1/N of parameters
- Use `state_dict_type="FULL_STATE_DICT"` for saving
- Automatic gathering on main rank

**Tensor Parallelism:**
- Parameters split across devices (e.g., column/row parallel)
- Must concatenate sharded tensors

'''Model Registry Pattern:'''

After export, models often uploaded to registries:
```
Model Repository (HuggingFace Hub, MLflow, etc.)
    ├── Model Card (description, metrics, usage)
    ├── Model Files (weights, config)
    ├── Example Code (inference examples)
    ├── Metadata (tags, framework version)
    └── Versioning (v1.0, v1.1, ...)
```

This enables:
* Discoverability: Others can find the model
* Reproducibility: Complete training config preserved
* Versioning: Track model iterations
* Collaboration: Share across teams

'''Compression and Optimization:'''

Post-training export optimizations:
* Quantization: FP32 → INT8 (4x smaller, faster inference)
* Pruning: Remove low-importance weights
* Distillation: Train smaller student model
* ONNX export: Framework-agnostic format for deployment

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_transformers_Trainer_save_model]]
