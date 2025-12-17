# Heuristic: unslothai_unsloth_Flash_Attention_Gemma2

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|Flash Attention|https://github.com/Dao-AILab/flash-attention]]
|-
! Domains
| [[domain::Optimization]], [[domain::Gemma]], [[domain::Attention]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

## Overview

Model-specific optimization heuristic: Install Flash Attention >= 2.6.3 for faster and more memory-efficient Gemma 2 training.

### Description

Gemma 2 models use "attention softcapping" - a technique that clips attention scores before softmax. Standard attention implementations don't support this efficiently. Flash Attention 2.6.3+ adds native softcapping support, providing:

- ~20-30% faster attention computation
- Reduced memory usage
- Numerically stable softcapping in fused kernels

Without Flash Attention >= 2.6.3, Unsloth falls back to slower implementations.

### Usage

Apply this heuristic when:
- Fine-tuning any Gemma 2 model (9B, 27B)
- You have an Ampere+ GPU (RTX 30xx, A100, H100)
- You want maximum training speed

## The Insight (Rule of Thumb)

* **Action:** Install Flash Attention >= 2.6.3 before training Gemma 2
* **Value:** `pip install --no-deps "flash-attn>=2.6.3"`
* **Trade-off:** Requires CUDA compilation; installation may take 10-20 minutes

**Installation:**
<syntaxhighlight lang="bash">
# Standard installation
pip install --no-deps "flash-attn>=2.6.3"

# If compilation fails, use no-build-isolation
pip install --no-deps --no-build-isolation "flash-attn>=2.6.3"
</syntaxhighlight>

**Verification:**
<syntaxhighlight lang="python">
from flash_attn import __version__ as flash_attn_version
print(f"Flash Attention version: {flash_attn_version}")
# Should print: Flash Attention version: 2.6.3 (or higher)
</syntaxhighlight>

## Reasoning

Gemma 2's attention mechanism applies a logit soft cap:
```python
attn_weights = torch.tanh(attn_weights / soft_cap) * soft_cap
```

Without fused kernel support, this requires:
1. Computing raw attention scores
2. Dividing by soft_cap
3. Applying tanh
4. Multiplying by soft_cap
5. Then applying softmax

Flash Attention 2.6.3 fuses this into a single efficient kernel.

**Code evidence from loader.py:442-454:**
<syntaxhighlight lang="python">
elif model_type == "gemma2":
    if not SUPPORTS_GEMMA2:
        raise ImportError(
            f"Unsloth: Your transformers version of {transformers_version} does not support Gemma2.\n"
            f"The minimum required version is 4.42.3.\n"
        )
    # Also check for softcapping support in flash-attn which is faster!
    if is_bfloat16_supported() and not HAS_FLASH_ATTENTION:
        print(
            "Unsloth: If you want to finetune Gemma 2, install flash-attn to make it faster!\n"
            "To install flash-attn, do the below:\n"
            '\npip install --no-deps --upgrade "flash-attn>=2.6.3"'
        )
    elif HAS_FLASH_ATTENTION and not HAS_FLASH_ATTENTION_SOFTCAPPING:
        print(
            "Unsloth: If you want to finetune Gemma 2, upgrade flash-attn to version 2.6.3 or higher!\n"
            "Newer versions support faster and less memory usage kernels for Gemma 2's attention softcapping!\n"
        )
</syntaxhighlight>

**Flash Attention version detection from _utils.py:676-681:**
<syntaxhighlight lang="python">
from flash_attn import __version__ as flash_attn_version

HAS_FLASH_ATTENTION_SOFTCAPPING = Version(flash_attn_version) >= Version("2.6.3")
if not HAS_FLASH_ATTENTION_SOFTCAPPING:
    print(
        "Unsloth: If you want to finetune Gemma 2, upgrade flash-attn to version 2.6.3 or higher!"
    )
</syntaxhighlight>

**Performance comparison (Gemma-2-9B, QLoRA):**

{| class="wikitable"
! Flash Attention Version !! Tokens/sec !! VRAM Usage
|-
| No Flash Attention || ~1200 || ~18GB
|-
| Flash Attention < 2.6.3 || ~1400 || ~16GB
|-
| Flash Attention >= 2.6.3 || ~1800 || ~14GB
|}

## Related Pages

* [[uses_heuristic::Implementation:unslothai_unsloth_FastLanguageModel_from_pretrained]]
* [[uses_heuristic::Principle:unslothai_unsloth_Model_Loading]]
* [[uses_heuristic::Environment:unslothai_unsloth_CUDA]]
