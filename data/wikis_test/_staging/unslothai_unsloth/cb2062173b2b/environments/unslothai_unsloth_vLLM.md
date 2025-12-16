# Environment: unslothai_unsloth_vLLM

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|vLLM Documentation|https://docs.vllm.ai/]]
|-
! Domains
| [[domain::Inference]], [[domain::Reinforcement_Learning]], [[domain::GRPO]]
|-
! Last Updated
| [[last_updated::2025-12-16 12:00 GMT]]
|}

## Overview

vLLM installation for fast inference during GRPO/RL training and for `fast_inference=True` model loading.

### Description

vLLM provides high-throughput inference for Unsloth in two scenarios:

1. **GRPO Training**: Efficient batch generation during reinforcement learning with group-relative policy optimization
2. **Fast Inference Mode**: Enabled via `fast_inference=True` in `FastLanguageModel.from_pretrained()`

vLLM enables:
- PagedAttention for efficient KV cache management
- Continuous batching for high throughput
- FP8 quantization support (requires `fast_inference=True`)
- Tensor parallelism for multi-GPU inference

### Usage

Required when:
- Setting `fast_inference=True` in model loading
- Using `load_in_fp8=True` (FP8 quantization)
- Running GRPO training with efficient generation

## System Requirements

{| class="wikitable"
! Category !! Requirement !! Notes
|-
| OS || Linux (Ubuntu 20.04+ recommended) || macOS/Windows not supported
|-
| Hardware || NVIDIA GPU || Ampere or newer recommended for best performance
|-
| CUDA || 11.8+ || Required for vLLM compilation
|-
| RAM || 32GB+ || For efficient CPU offloading
|-
| VRAM || 16GB+ || Recommended for 7B models with vLLM
|}

## Dependencies

### System Packages

* `cuda-toolkit` >= 11.8
* `gcc` >= 9.0

### Python Packages

* `vllm` >= 0.4.0
* `torch` >= 2.0.0
* `ray` (for distributed inference, optional)

### Installation

```bash
pip install vllm
```

## Credentials

* `HF_TOKEN`: For accessing gated models during inference

## Code Evidence

From `unsloth/models/loader.py:209-214`:
```python
if fast_inference:
    if importlib.util.find_spec("vllm") is None:
        raise ImportError(
            "Unsloth: Please install vLLM before enabling `fast_inference`!\n"
            "You can do this in a terminal via `pip install vllm`"
        )
```

From `unsloth/models/loader.py:216-220`:
```python
# [TODO] For now fast_inference only works with fast_inference ie vLLM
if load_in_fp8 != False:
    if not fast_inference:
        raise NotImplementedError(
            "Unsloth: set `fast_inference = True` when doing `load_in_fp8`."
        )
```

vLLM parameters in `FastLanguageModel.from_pretrained()`:
```python
fast_inference = False,  # uses vLLM
gpu_memory_utilization = 0.5,
float8_kv_cache = False,
max_lora_rank = 64,
disable_log_stats = True,
```

From `unsloth/models/rl.py:53-58` (vLLM SamplingParams):
```python
def vLLMSamplingParams(**kwargs):
    from vllm import SamplingParams
    sampling_params = SamplingParams(**kwargs)
    sampling_params._set_kwargs = kwargs
    return sampling_params
```

## Related Pages

* [[requires_env::Implementation:unslothai_unsloth_FastLanguageModel]]
* [[requires_env::Workflow:unslothai_unsloth_GRPO_Training]]
