# Heuristic: AMD GPU Compatibility and Optimization

## Category
Hardware/Compatibility

## Summary
Unsloth provides experimental AMD GPU (ROCm/HIP) support with specific considerations for model loading, prequantized model compatibility, and performance optimization. This heuristic guides AMD users through the compatibility landscape.

## The Decision Framework

### AMD Support Status

| Feature | Status | Notes |
|---------|--------|-------|
| Basic training | Supported | Via ROCm/HIP |
| 4-bit quantization | Limited | Prequantized models restricted |
| Flash Attention | Partial | Depends on ROCm version |
| bfloat16 | Hardware dependent | Check GPU capability |

## Implementation Evidence

From `unsloth/device_type.py`:

```python
# Device type detection
DEVICE_TYPE = "cuda"  # Default
if torch.cuda.is_available():
    if hasattr(torch.version, 'hip') and torch.version.hip:
        DEVICE_TYPE = "hip"  # AMD ROCm
```

### Prequantized Model Restrictions

From `device_type.py`:

```python
# AMD GPUs have limitations with some prequantized models
ALLOW_PREQUANTIZED_MODELS = True
if DEVICE_TYPE == "hip":
    ALLOW_PREQUANTIZED_MODELS = False  # Disabled for AMD
```

## AMD-Specific Considerations

### ROCm Requirements

| Component | Minimum Version | Recommended |
|-----------|-----------------|-------------|
| ROCm | 5.4+ | 6.0+ |
| PyTorch (ROCm) | 2.1.0 | 2.3.0+ |

### GPU Compatibility

| AMD GPU Series | Support Level | Notes |
|----------------|--------------|-------|
| RX 7900 XTX | Good | Full ROCm support |
| RX 6900 XT | Partial | Some limitations |
| MI250/MI300 | Best | Data center GPUs |
| Older GPUs | Limited | May not work |

## Tribal Knowledge

### What Works on AMD

1. **Training from scratch models**: Full compatibility
2. **LoRA fine-tuning**: Works with caveats
3. **16-bit models**: Generally supported
4. **HuggingFace integration**: Through PyTorch ROCm

### What Doesn't Work (or has issues)

1. **Prequantized 4-bit models**: Often incompatible
2. **CUDA-specific kernels**: Need ROCm equivalents
3. **Some Flash Attention versions**: Check compatibility
4. **FP8 quantization**: Not supported on AMD

## Decision Tree

```
Start
  │
  ├─ Is GPU AMD (ROCm/HIP)?
  │   └─ Check: torch.version.hip exists
  │
  ├─ ROCm version >= 5.4?
  │   ├─ No → Upgrade ROCm
  │   └─ Yes → Continue
  │
  ├─ Want to use prequantized model?
  │   ├─ Yes → Convert to 16-bit first, then quantize
  │   └─ No → Proceed normally
  │
  └─ Training issues?
      ├─ Try disabling Flash Attention
      └─ Use float16 instead of bfloat16
```

## Configuration Workarounds

### For AMD Users

```python
# Disable prequantized model loading
import os
os.environ["UNSLOTH_ALLOW_PREQUANTIZED"] = "0"

# Load model
model = FastLanguageModel.from_pretrained(
    model_name,
    load_in_4bit=False,  # Load in 16-bit first
    dtype=torch.float16,  # Use float16 for better compatibility
)
```

### Manual Quantization for AMD

Instead of using prequantized models:

```python
# 1. Load model in 16-bit
model = FastLanguageModel.from_pretrained(
    model_name,
    load_in_4bit=False,
)

# 2. Quantize after loading (if needed)
# This avoids prequantized model compatibility issues
```

## Environment Variables

| Variable | Purpose | AMD Default |
|----------|---------|-------------|
| `HIP_VISIBLE_DEVICES` | GPU selection | All AMD GPUs |
| `ROCM_HOME` | ROCm installation path | Auto-detected |
| `UNSLOTH_ALLOW_PREQUANTIZED` | Enable prequantized | `0` for AMD |

## Common Issues and Solutions

### Issue 1: Prequantized Model Fails
**Solution**: Load as 16-bit, quantize yourself

### Issue 2: Training Crashes
**Solution**:
- Reduce batch size
- Disable Flash Attention
- Use float16 instead of bfloat16

### Issue 3: Slow Performance
**Solution**:
- Ensure ROCm is properly installed
- Check `rocm-smi` for GPU utilization
- Update to latest PyTorch ROCm build

## Source Evidence

- Device Detection: `unsloth/device_type.py`
- Model Loading: `unsloth/models/loader.py`
- Hardware Checks: `unsloth/models/_utils.py`

## Backlinks

[[used_by::Implementation:Unslothai_Unsloth_FastLanguageModel_from_pretrained]]
[[used_by::Implementation:Unslothai_Unsloth_Device_Type]]

## Related

- [[Environment:Unslothai_Unsloth_CUDA_11]]
- [[Heuristic:Unslothai_Unsloth_Gradient_Checkpointing]]
