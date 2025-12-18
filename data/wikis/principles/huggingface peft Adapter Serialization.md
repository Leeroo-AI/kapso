{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Doc|Safetensors|https://huggingface.co/docs/safetensors]]
|-
! Domains
| [[domain::Serialization]], [[domain::Adapter]], [[domain::Persistence]]
|-
! Last Updated
| [[last_updated::2025-01-15 12:00 GMT]]
|}

== Overview ==

Principle for saving trained adapter weights independently of the base model for efficient storage and sharing.

=== Description ===

Adapter Serialization saves only the trained adapter parameters, not the full model. This results in dramatically smaller checkpoint sizes—typically 10-100MB for adapters vs. multiple GB for full models. The serialized adapter includes:
1. Adapter weights (in safetensors or PyTorch format)
2. Configuration (as JSON)
3. Optional model card for documentation

This enables efficient sharing, version control, and deployment of task-specific adaptations.

=== Usage ===

Apply this principle after training completes:
* **Standard save:** Use `model.save_pretrained()` for local storage
* **Hub upload:** Use `model.push_to_hub()` for sharing
* **Multi-adapter:** Use `selected_adapters` to save specific adapters
* **Compatibility:** Enable `safe_serialization=True` for safetensors format

== Theoretical Basis ==

'''Selective State Dict:'''

PEFT extracts only adapter-related weights:
<syntaxhighlight lang="python">
# Pseudo-code for adapter state dict extraction
def get_peft_model_state_dict(model, adapter_name):
    state_dict = {}

    for name, param in model.named_parameters():
        if should_save(name, adapter_name):
            # Include: lora_A, lora_B, modules_to_save
            state_dict[name] = param.data
        # Exclude: base model weights (not saved)

    return state_dict
</syntaxhighlight>

'''Storage Efficiency:'''

For a 7B parameter model with LoRA (r=16) on attention layers:
* Base model: ~14 GB (float16)
* LoRA adapter: ~40 MB

Savings ratio:
<math>\text{Reduction} = 1 - \frac{\text{adapter size}}{\text{model size}} \approx 99.7\%</math>

'''Safetensors Format:'''

Safetensors provides:
* Security: No arbitrary code execution
* Speed: Memory-mapped loading
* Compatibility: Cross-framework support

'''Configuration Persistence:'''

The adapter config JSON includes all hyperparameters:
<syntaxhighlight lang="json">
{
  "peft_type": "LORA",
  "r": 16,
  "lora_alpha": 32,
  "target_modules": ["q_proj", "v_proj"],
  "lora_dropout": 0.05,
  "bias": "none",
  "task_type": "CAUSAL_LM",
  "base_model_name_or_path": "meta-llama/Llama-2-7b-hf"
}
</syntaxhighlight>

This enables automatic reconstruction when loading.

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_peft_PeftModel_save_pretrained]]
