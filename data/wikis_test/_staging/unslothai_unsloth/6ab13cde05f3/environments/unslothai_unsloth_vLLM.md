# Environment: unslothai_unsloth_vLLM

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Doc|rl.py|unsloth/models/rl.py]]
* [[source::Doc|loader.py|unsloth/models/loader.py]]
|-
! Domains
| [[domain::Infrastructure]], [[domain::Reinforcement_Learning]], [[domain::Inference]]
|-
! Last Updated
| [[last_updated::2025-12-16 18:00 GMT]]
|}

## Overview

vLLM inference engine environment for fast sampling during GRPO/PPO reinforcement learning training, enabling efficient generation with PagedAttention.

### Description

This environment provides vLLM integration for fast inference during RL training workflows. vLLM enables efficient text generation using PagedAttention, which is critical for GRPO and PPO where large numbers of completions must be generated per training step.

Key features:
- **Colocate Mode**: vLLM runs on the same GPU as training
- **PagedAttention**: Memory-efficient KV cache management
- **LoRA Support**: Dynamic LoRA adapter loading during inference
- **Sampling Parameters**: Configurable temperature, top-k, top-p sampling

### Usage

Use this environment for:
- **GRPO Training**: `PatchFastRL()` enables vLLM-accelerated generation
- **Fast Inference**: `fast_inference=True` in model loading
- **RL Workflows**: Any TRL-based RL trainer (GRPOTrainer, PPOTrainer, etc.)

Required when:
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B-Instruct",
    fast_inference=True,  # Enables vLLM
    gpu_memory_utilization=0.5,
)
```

## System Requirements

{| class="wikitable"
! Category !! Requirement !! Notes
|-
| OS || Linux (Ubuntu 20.04+) || vLLM has limited Windows/Mac support
|-
| Hardware || NVIDIA GPU (Compute >= 7.0) || Ampere+ recommended for best performance
|-
| VRAM || 16GB+ || 24GB recommended for 7B models with vLLM overhead
|-
| CUDA || 11.8+ || Required for vLLM CUDA kernels
|}

## Dependencies

### System Packages

* `cuda-toolkit` >= 11.8
* `gcc`, `g++` >= 9

### Python Packages

* `vllm` >= 0.4.0
* `torch` >= 2.1.0
* `trl` >= 0.18.0 (for vLLM colocate mode)
* `ray` (optional, for distributed inference)

### Version Checks

From `loader.py:209-214`:
```python
if fast_inference:
    if importlib.util.find_spec("vllm") is None:
        raise ImportError(
            "Unsloth: Please install vLLM before enabling `fast_inference`!\n"
            "You can do this in a terminal via `pip install vllm`"
        )
```

## Credentials

* `HF_TOKEN`: For loading gated models with vLLM

## Configuration Options

Key parameters for vLLM setup:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fast_inference` | False | Enable vLLM inference engine |
| `gpu_memory_utilization` | 0.5 | Fraction of GPU memory for vLLM |
| `max_lora_rank` | 64 | Maximum LoRA rank for vLLM adapter |
| `float8_kv_cache` | False | Use FP8 for KV cache compression |
| `disable_log_stats` | True | Suppress vLLM logging |

## Code Evidence

vLLM sampling params from `rl.py:53-58`:
```python
def vLLMSamplingParams(**kwargs):
    from vllm import SamplingParams
    sampling_params = SamplingParams(**kwargs)
    sampling_params._set_kwargs = kwargs
    return sampling_params
```

vLLM colocate mode from `rl.py:1097-1107`:
```python
if "grpo" in trainer_file and trl_version >= Version("0.18.0"):
    vllm_setter += " " * 12 + "args.vllm_mode='colocate'\n"
    if trl_version >= Version("0.23.0"):
        vllm_setter += (
            " " * 12
            + "if os.environ.get('UNSLOTH_VLLM_STANDBY', '0') == '1':\n"
            + " " * 16
            + "args.vllm_enable_sleep_mode=True\n"
        )
```

LoRA request injection from `rl.py:1265-1276`:
```python
source = re.sub(
    r"(self\.llm\.(?:generate|chat)\([^\)]{1,})\)",
    r"\1, lora_request = self.model.load_lora('"
    + lora_name
    + r", load_tensors = True))",
    source,
)
```

## Related Pages

* [[requires_env::Implementation:unslothai_unsloth_PatchFastRL]]
