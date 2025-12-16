# Environment: unslothai_unsloth_Storage

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|HuggingFace Safetensors|https://huggingface.co/docs/safetensors/]]
|-
! Domains
| [[domain::Infrastructure]], [[domain::Model_Export]]
|-
! Last Updated
| [[last_updated::2025-12-16 12:00 GMT]]
|}

## Overview

Storage requirements for model saving, merging, and GGUF conversion including temporary buffers and sharded output files.

### Description

Unsloth's model saving pipeline requires significant disk space for:

1. **Temporary Buffers**: Layer-by-layer dequantization and merging (`_unsloth_temporary_saved_buffers/`)
2. **Merged Model**: Full 16-bit model weights before quantization
3. **GGUF Output**: Quantized model files in various formats
4. **Sharded Safetensors**: For models larger than `max_shard_size` (default 5GB)

The storage is environment-specific:
- **Colab**: Uses standard /content directory
- **Kaggle**: Limited to 20GB, must use /tmp for large models
- **Local**: Requires planning based on model size

### Usage

Required for:
- `model.save_pretrained()` - Saving LoRA or merged models
- `model.save_pretrained_merged()` - Merging LoRA into base weights
- `model.save_pretrained_gguf()` - GGUF conversion pipeline

## System Requirements

{| class="wikitable"
! Category !! Requirement !! Notes
|-
| Disk Type || SSD recommended || High IOPS for temporary file operations
|-
| Free Space || Model Size x 3 || Base + merged + GGUF output
|-
| Colab || Standard quota || Usually sufficient for 7B models
|-
| Kaggle || 20GB limit || Use /tmp for large operations
|}

### Space Estimates by Model Size

{| class="wikitable"
! Model Size !! 16-bit Merged !! GGUF q4_k_m !! Total Needed
|-
| 1B params || ~2GB || ~0.5GB || ~6GB
|-
| 7B params || ~14GB || ~4GB || ~35GB
|-
| 13B params || ~26GB || ~7GB || ~65GB
|-
| 70B params || ~140GB || ~40GB || ~350GB
|}

## Dependencies

### Python Packages

* `safetensors` >= 0.4.0 (for efficient model serialization)
* `torch` >= 2.0.0 (for tensor operations)

### Environment Variables

* `KAGGLE_TMP`: Set to `/tmp` in Kaggle environments

## Credentials

For pushing to HuggingFace Hub:
* `HF_TOKEN`: HuggingFace write token for upload

## Code Evidence

From `unsloth/save.py:78-80`:
```python
IS_COLAB_ENVIRONMENT = "\nCOLAB_" in keynames
IS_KAGGLE_ENVIRONMENT = "\nKAGGLE_" in keynames
KAGGLE_TMP = "/tmp"
```

From `unsloth/save.py:251-253`:
```python
temporary_location: str = "_unsloth_temporary_saved_buffers",
maximum_memory_usage: float = 0.9,
# ...
assert maximum_memory_usage > 0 and maximum_memory_usage <= 0.95
```

Memory-efficient layer processing from `unsloth/save.py:252`:
```python
maximum_memory_usage: float = 0.9,  # Controls GPU memory usage during merging
```

## Related Pages

* [[requires_env::Implementation:unslothai_unsloth_save_pretrained_merged]]
