# Principle: unslothai_unsloth_Model_Saving

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|HuggingFace Hub|https://huggingface.co/docs/hub]]
* [[source::Doc|SafeTensors Format|https://huggingface.co/docs/safetensors]]
* [[source::Paper|LoRA: Low-Rank Adaptation|https://arxiv.org/abs/2106.09685]]
|-
! Domains
| [[domain::NLP]], [[domain::Model_Export]], [[domain::Serialization]]
|-
! Last Updated
| [[last_updated::2025-12-17 15:00 GMT]]
|}

== Overview ==

Technique for persisting trained models in formats suitable for deployment, sharing, or further processing.

=== Description ===

Model Saving after QLoRA training involves decisions about:

1. **What to save**: LoRA adapters only vs merged weights
2. **Precision**: 16-bit (recommended) vs 4-bit
3. **Format**: HuggingFace safetensors, GGUF for llama.cpp, etc.
4. **Destination**: Local disk vs HuggingFace Hub

For QLoRA models, the key operation is **LoRA merging**: combining the low-rank adapter weights with the frozen base weights to produce a standalone model.

=== Usage ===

Use this principle when:
- Exporting a trained model for deployment
- Preparing a model for format conversion (e.g., GGUF)
- Sharing a model on HuggingFace Hub
- Creating checkpoints during long training runs

== Theoretical Basis ==

=== LoRA Merging ===

The merge operation combines base weights with LoRA adapters:

<math>
W_{merged} = W_{base} + \frac{\alpha}{r} \cdot B \cdot A
</math>

Where:
- <math>W_{base}</math>: Original frozen weights (4-bit quantized)
- <math>A, B</math>: Trained LoRA matrices
- <math>\alpha/r</math>: Scaling factor

'''Pseudo-code for merge:'''
<syntaxhighlight lang="python">
def merge_lora_weights(base_weight_4bit, A, B, alpha, rank):
    # 1. Dequantize base weights from 4-bit to float
    base_weight = dequantize_nf4(base_weight_4bit)  # [d, k]

    # 2. Compute LoRA contribution
    lora_contribution = (alpha / rank) * (B @ A)  # [d, k]

    # 3. Merge
    merged_weight = base_weight + lora_contribution

    # 4. Cast to target precision
    return merged_weight.to(torch.float16)
</syntaxhighlight>

=== Dequantization Process ===

4-bit NF4 weights must be dequantized before merging:

<syntaxhighlight lang="python">
def dequantize_nf4(quantized, quant_state):
    """
    Convert 4-bit quantized weights back to float.

    quantized: int4 indices into NF4 codebook
    quant_state: Contains absmax scaling factors
    """
    # NF4 codebook (16 values)
    NF4_CODEBOOK = torch.tensor([
        -1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
        0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0
    ])

    # Map indices to values
    dequantized = NF4_CODEBOOK[quantized]

    # Scale by block absmax
    dequantized = dequantized * quant_state.absmax

    return dequantized.reshape(original_shape)
</syntaxhighlight>

=== Save Format Comparison ===

{| class="wikitable"
|-
! Format !! File Extension !! Use Case !! Notes
|-
| SafeTensors || `.safetensors` || HuggingFace ecosystem || Fast, safe, recommended
|-
| PyTorch || `.bin` || Legacy compatibility || Slower, pickle-based
|-
| GGUF || `.gguf` || llama.cpp, Ollama || CPU inference
|-
| LoRA only || `adapter_model.safetensors` || Lightweight sharing || Requires base model
|}

=== Numerical Precision ===

When merging and saving:

<syntaxhighlight lang="python">
# Precision cascade during merge
# Input: 4-bit (base) + 16-bit (LoRA)
# Compute: float32 (for numerical stability)
# Output: float16/bfloat16 (storage)

def merge_with_precision(base_4bit, lora_A, lora_B):
    # Upcast for computation
    base_f32 = dequantize_to_f32(base_4bit)

    # LoRA in float32
    lora_contribution = (lora_B.float() @ lora_A.float())

    # Merge in float32
    merged_f32 = base_f32 + lora_contribution

    # Downcast for storage
    return merged_f32.to(torch.float16)  # or bfloat16
</syntaxhighlight>

=== Sharding for Large Models ===

Large models are split into shards:

<syntaxhighlight lang="python">
# Sharding logic
def shard_model(state_dict, max_shard_size="5GB"):
    max_bytes = parse_size(max_shard_size)  # 5 * 1024^3
    current_shard = {}
    current_size = 0
    shards = []

    for key, tensor in state_dict.items():
        tensor_size = tensor.numel() * tensor.element_size()

        if current_size + tensor_size > max_bytes:
            # Start new shard
            shards.append(current_shard)
            current_shard = {}
            current_size = 0

        current_shard[key] = tensor
        current_size += tensor_size

    shards.append(current_shard)  # Last shard
    return shards
</syntaxhighlight>

== Practical Guide ==

=== Choosing Save Method ===

| Scenario | Recommended Method |
|----------|-------------------|
| Deploy with HuggingFace | `save_pretrained_merged` → Hub |
| Convert to GGUF | `save_pretrained_merged` locally first |
| Share LoRA only | `save_pretrained` (adapters only) |
| Checkpoint during training | Trainer handles automatically |

=== Memory During Saving ===

Saving requires temporary memory for dequantization:

<syntaxhighlight lang="python">
# Memory estimate for 7B model
base_model_4bit = 7e9 * 0.5 / 1e9  # ~3.5 GB
dequantized_f32 = 7e9 * 4 / 1e9    # ~28 GB (temporary)
saved_f16 = 7e9 * 2 / 1e9          # ~14 GB

# Peak memory ≈ 3.5 + 28 = 31.5 GB
# May need to process in chunks for limited VRAM
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:unslothai_unsloth_save_pretrained_merged]]

=== Used In Workflows ===
* [[used_by::Workflow:unslothai_unsloth_QLoRA_Finetuning]]
* [[used_by::Workflow:unslothai_unsloth_GRPO_Reinforcement_Learning]]
* [[used_by::Workflow:unslothai_unsloth_Model_Export]]
