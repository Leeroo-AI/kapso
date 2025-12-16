# Heuristic: unslothai_unsloth_Mixed_Precision_Training

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|rl.py|unsloth/models/rl.py]]
* [[source::Doc|loader.py|unsloth/models/loader.py]]
|-
! Domains
| [[domain::Optimization]], [[domain::LLMs]], [[domain::Training]]
|-
! Last Updated
| [[last_updated::2025-12-16 18:00 GMT]]
|}

## Overview

Automatic mixed precision (AMP) configuration for Unsloth training, handling fp16/bf16 selection based on model dtype and hardware support.

### Description

Unsloth automatically configures mixed precision training based on the loaded model's dtype. This prevents dtype mismatch errors and optimizes for the available hardware. The system detects whether a model uses float16 or bfloat16 and sets training arguments accordingly.

Key features:
- **Auto-detection**: Reads model dtype from config
- **Validation**: Prevents fp16 model + bf16 training mismatch
- **Fallback**: Supports float32 training when required
- **Environment override**: Can force specific precision via env vars

### Usage

Apply this heuristic when:
- Encountering dtype mismatch errors during training
- Optimizing for specific hardware (A100 prefers bf16, older GPUs use fp16)
- Debugging NaN/Inf losses that may be precision-related
- Setting up training on non-NVIDIA hardware

## The Insight (Rule of Thumb)

### Automatic Precision Selection

Unsloth auto-configures based on model dtype:
- **Model in float16** → Set `fp16=True, bf16=False`
- **Model in bfloat16** → Set `bf16=True, fp16=False`
- **Force float32** → Set both to False (via env var)

### Environment Variables

| Variable | Values | Effect |
|----------|--------|--------|
| `UNSLOTH_FORCE_FLOAT32` | `0`/`1` | Force float32 training |
| `UNSLOTH_MIXED_PRECISION` | `float32`/`bfloat16` | Override precision |
| `ACCELERATE_MIXED_PRECISION` | `no`/`fp16`/`bf16` | Accelerate config |

### Common Errors and Solutions

**Error**: "Model is in float16 but you want to use bfloat16 precision"
**Solution**: Set `fp16=True, bf16=False` in TrainingArguments OR let Unsloth auto-configure

**Error**: "Model is in bfloat16 but you want to use float16 precision"
**Solution**: Set `bf16=True, fp16=False` in TrainingArguments OR let Unsloth auto-configure

### Evaluation Precision

For consistency, evaluation uses same precision as training:
```python
if args.fp16 and bf16_full_eval:
    args.bf16_full_eval = False
    args.fp16_full_eval = True
if args.bf16 and fp16_full_eval:
    args.bf16_full_eval = True
    args.fp16_full_eval = False
```

### Full Fine-tuning

For non-LoRA full fine-tuning with bfloat16:
- Both `fp16` and `bf16` are set to False
- Uses native bfloat16 without autocasting
- Set via `UNSLOTH_MIXED_PRECISION=bfloat16`

## Reasoning

**Why auto-detection?**
Mixed precision works by using lower precision (fp16/bf16) for forward/backward passes while keeping master weights in fp32. The accumulation dtype must match the model's compute dtype for numerical stability.

**Why bfloat16 for modern GPUs?**
- Ampere+ GPUs have native bf16 tensor cores
- bf16 has same exponent range as fp32, avoiding overflow
- fp16 can overflow on large activations, requiring loss scaling

**Why the dtype mismatch error?**
If a model was trained in fp16 and you run with bf16 autocast, the weight/activation dtypes conflict, causing:
- Incorrect gradient computation
- Potential NaN losses
- Numerical instability

## Code Evidence

Dtype detection and validation from `rl.py:450-491`:
```python
mixed_precision = (
    "use_bf16 = getattr(args, 'bf16', False)\n"
    "if type(use_bf16) is not bool: use_bf16 = False\n"
    "use_fp16 = getattr(args, 'fp16', False)\n"
    "if type(use_fp16) is not bool: use_fp16 = False\n"
    "force_float32 = False\n"
    "full_finetuning = os.environ.get('UNSLOTH_ENABLE_FULL_FINETUNING', '0') == '1'\n"
    "if not full_finetuning and (os.environ.get('UNSLOTH_FORCE_FLOAT32', '0') == '1'):\n"
    "    print('Unsloth: Switching to float32 training since model cannot work with float16')\n"
    "    force_float32 = True\n"
    "mixed_precision_dtype = os.environ.get('UNSLOTH_MIXED_PRECISION', 'float32')\n"
    "dtype = getattr(model.config, 'dtype', None) or getattr(model.config, 'torch_dtype', None)\n"
    "if dtype is None: dtype = model.get_input_embeddings().weight.dtype\n"
    "from unsloth_zoo.utils import _get_dtype\n"
    "dtype = _get_dtype(dtype)\n"
    "float16 = dtype == torch.float16\n"
    "if not force_float32 and (float16 and use_bf16): raise TypeError('Unsloth: Model is in float16 precision but you want to use bfloat16 precision. Set fp16 to `True` and bf16 to `False`')\n"
    "if not force_float32 and (not float16 and use_fp16): raise TypeError('Unsloth: Model is in bfloat16 precision but you want to use float16 precision. Set fp16 to `False` and bf16 to `True`')\n"
)
```

Automatic precision setup from `rl.py:474-488`:
```python
"elif (not use_bf16 and not use_fp16) and mixed_precision_dtype == 'float32':\n"
"    # Mixed precision training\n"
"    args.fp16 = float16\n"
"    args.bf16 = not float16\n"
"    os.environ['ACCELERATE_MIXED_PRECISION'] = 'fp16' if float16 else 'bf16'\n"
"    if hasattr(args, 'mixed_precision'): args.mixed_precision = 'fp16' if float16 else 'bf16'\n"
"elif mixed_precision_dtype == 'bfloat16':\n"
"    # Both False since bfloat16 full finetuning doesn't do any autocasting.\n"
"    args.fp16 = False\n"
"    args.bf16 = False\n"
"    os.environ['ACCELERATE_MIXED_PRECISION'] = 'no'\n"
```

## Related Pages

* [[uses_heuristic::Implementation:unslothai_unsloth_FastLanguageModel]]
* [[uses_heuristic::Workflow:unslothai_unsloth_QLoRA_Finetuning]]
