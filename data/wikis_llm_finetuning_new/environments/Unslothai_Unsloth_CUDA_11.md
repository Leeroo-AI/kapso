# Environment: CUDA 11+ with Compute Capability 8.0+

## Category
Hardware/GPU

## Summary
Unsloth requires NVIDIA GPUs with CUDA 11+ and compute capability 8.0 or higher for optimal performance, enabling bfloat16 support and Flash Attention acceleration.

## Requirements

### Hardware Requirements
| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| GPU Architecture | Ampere (SM 8.0) | Hopper (SM 9.0) | Required for bfloat16 and Flash Attention |
| CUDA Compute Capability | 8.0 | 8.9+ | Checked at runtime via `torch.cuda.get_device_capability()` |
| VRAM | 8GB | 16GB+ | Depends on model size |

### Software Requirements
| Package | Version Constraint | Evidence |
|---------|-------------------|----------|
| CUDA Toolkit | >= 11.0 | Implicit via PyTorch CUDA support |
| PyTorch | >= 2.1.0 | Required for CUDA support |
| Flash Attention | >= 2.6.3 | Required for softcapping support in Gemma2/Cohere |

### Environment Variables
| Variable | Purpose | Default |
|----------|---------|---------|
| `CUDA_VISIBLE_DEVICES` | GPU selection | All GPUs |
| `UNSLOTH_DISABLE_STATISTICS` | Disable telemetry | `0` |

## Detection Logic

From `unsloth/models/_utils.py`:

```python
major_version, minor_version = torch.cuda.get_device_capability()
SUPPORTS_BFLOAT16 = False

if major_version >= 8:
    SUPPORTS_BFLOAT16 = True
    if _is_package_available("flash_attn"):
        HAS_FLASH_ATTENTION = True
```

The system checks GPU compute capability at runtime to determine:
1. **bfloat16 support**: Requires compute capability >= 8.0 (Ampere+)
2. **Flash Attention availability**: Only enabled on supported hardware
3. **FP8 support**: Requires compute capability >= 8.9 (Ada Lovelace/Hopper)

## Compatible GPU Models

| GPU Family | Compute Capability | bfloat16 | Flash Attention | FP8 |
|------------|-------------------|----------|-----------------|-----|
| RTX 30xx (Ampere) | 8.6 | Yes | Yes | No |
| RTX 40xx (Ada) | 8.9 | Yes | Yes | Yes |
| A100 | 8.0 | Yes | Yes | No |
| H100 | 9.0 | Yes | Yes | Yes |
| L4 | 8.9 | Yes | Yes | Yes |

## Fallback Behavior

When compute capability < 8.0:
- bfloat16 disabled, falls back to float16
- Flash Attention may not be available
- Performance degradation expected

From `_utils.py:117-130`:
```python
if not SUPPORTS_BFLOAT16:
    SUPPORTS_BFLOAT16 = torch.cuda.is_bf16_supported()
```

## Source Evidence

- Primary: `unsloth/models/_utils.py:107-145`
- Device Detection: `unsloth/device_type.py:1-50`
- Model Loading: `unsloth/models/loader.py`

## Backlinks

[[required_by::Implementation:Unslothai_Unsloth_FastLanguageModel_from_pretrained]]
[[required_by::Implementation:Unslothai_Unsloth_get_peft_model]]
[[required_by::Implementation:Unslothai_Unsloth_FP8_Kernels]]

## Related

- [[Environment:Unslothai_Unsloth_AMD_GPU_Compatibility]]
- [[Environment:Unslothai_Unsloth_Vision]]
