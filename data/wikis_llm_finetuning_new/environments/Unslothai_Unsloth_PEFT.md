# Environment: PEFT (Parameter-Efficient Fine-Tuning) Integration

## Category
Software/Training

## Summary
Unsloth provides deep integration with Hugging Face PEFT library for LoRA fine-tuning, with version-specific patches and optimizations for memory-efficient training.

## Requirements

### Software Requirements
| Package | Version Constraint | Evidence |
|---------|-------------------|----------|
| PEFT | >= 0.7.0 | Core LoRA functionality |
| PEFT | < 0.12.0 | Requires patching for older versions |
| bitsandbytes | >= 0.41.0 | 4-bit quantization support |

### Version Compatibility

| PEFT Version | Behavior | Notes |
|--------------|----------|-------|
| < 0.12.0 | Requires patching | Older API compatibility |
| >= 0.12.0 | Native support | Modern LoRA implementation |

## LoRA Integration

From `unsloth/save.py:26-28`:

```python
from bitsandbytes.nn import Linear4bit as Bnb_Linear4bit
from peft.tuners.lora import Linear4bit as Peft_Linear4bit
from peft.tuners.lora import Linear as Peft_Linear
```

### LoRA Merging

From `save.py:189-218`:

```python
def _merge_lora(layer, name):
    bias = getattr(layer, "bias", None)
    if isinstance(layer, (Bnb_Linear4bit, Peft_Linear4bit, Peft_Linear)):
        # Is LoRA so we need to merge!
        W, quant_state, A, B, s, bias = get_lora_parameters_bias(layer)
        if quant_state is not None:
            dtype = (
                quant_state.dtype if type(quant_state) is not list else quant_state[2]
            )
            W = fast_dequantize(W, quant_state)
        else:
            dtype = W.dtype
        W = W.to(torch.float32).t()

        if A is not None:
            W.addmm_(A.t().to(torch.float32), B.t().to(torch.float32), alpha=s)
```

## PeftModel Detection

From `save.py:59,422-428`:

```python
from peft import PeftModelForCausalLM, PeftModel

# Check if PEFT Model or not - if yes, 3 levels. If not 2 levels.
if isinstance(model, PeftModelForCausalLM):
    internal_model = model.model
else:
    internal_model = model
```

## Layer Weight Patterns

From `save.py:84-100`:

```python
LLAMA_WEIGHTS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
LLAMA_LAYERNORMS = (
    "input_layernorm",
    "post_attention_layernorm",
    "pre_feedforward_layernorm",
    "post_feedforward_layernorm",
    "self_attn.q_norm",
    "self_attn.k_norm",
)
```

## Save Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| `lora` | Save LoRA adapters only | Smallest files, for continued training |
| `merged_16bit` | Merge and save as float16 | For GGUF conversion |
| `merged_4bit` | Merge and save as 4-bit | For inference with quantization |

From `save.py:239`:
```python
save_method: str = "lora",  # ["lora", "merged_16bit", "merged_4bit"]
```

## Memory Optimization

From `save.py:559-579`:

```python
# Switch to our fast saving modules if it's a slow PC!
n_cpus = psutil.cpu_count(logical=False)
if n_cpus is None:
    n_cpus = psutil.cpu_count()

if safe_serialization and (n_cpus <= 2):
    logger.warning_once(
        f"Unsloth: You have {n_cpus} CPUs. Using `safe_serialization` is 10x slower.\n"
        f"We shall switch to Pytorch saving..."
    )
    safe_serialization = False
    save_function = fast_save_pickle
```

## Source Evidence

- LoRA Imports: `unsloth/save.py:26-28`
- Merge Logic: `unsloth/save.py:189-218`
- PeftModel Check: `unsloth/save.py:422-428`
- Layer Patterns: `unsloth/save.py:84-100`

## Backlinks

[[required_by::Implementation:Unslothai_Unsloth_get_peft_model]]
[[required_by::Implementation:Unslothai_Unsloth_FastLanguageModel_from_pretrained]]
[[required_by::Implementation:Unslothai_Unsloth_save_pretrained_merged]]

## Related

- [[Environment:Unslothai_Unsloth_TRL]]
- [[Heuristic:Unslothai_Unsloth_LoRA_Rank_Selection]]
