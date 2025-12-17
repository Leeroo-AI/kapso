# Implementation: unslothai_unsloth_FastLanguageModel_from_pretrained

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|Unsloth Docs|https://docs.unsloth.ai]]
* [[source::Paper|QLoRA|https://arxiv.org/abs/2305.14314]]
|-
! Domains
| [[domain::NLP]], [[domain::Model_Loading]], [[domain::Quantization]]
|-
! Last Updated
| [[last_updated::2025-12-17 15:00 GMT]]
|}

== Overview ==

Concrete tool for loading Large Language Models with 4-bit NF4 quantization for memory-efficient QLoRA fine-tuning.

=== Description ===

`FastLanguageModel.from_pretrained` is the primary entry point for loading models in Unsloth. It automatically:

1. **Detects model architecture** (Llama, Mistral, Qwen, Gemma, etc.) and dispatches to optimized loaders
2. **Applies 4-bit quantization** using bitsandbytes NF4 format for ~75% memory reduction
3. **Patches attention mechanisms** with optimized Triton kernels
4. **Configures RoPE embeddings** for extended context support
5. **Handles tokenizer fixes** for consistent behavior

This implementation focuses on the **QLoRA fine-tuning** use case where `load_in_4bit=True` provides maximum memory efficiency.

=== Usage ===

Use this when:
- Starting a QLoRA fine-tuning workflow
- Loading pre-quantized models from HuggingFace Hub
- Setting up memory-efficient training on consumer GPUs (RTX 3090, 4090, etc.)

NOT for:
- RL training with vLLM (use `FastLanguageModel_from_pretrained_vllm` instead)
- Full-precision fine-tuning (use `FastModel.from_pretrained`)
- Inference-only deployment

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unsloth]
* '''File:''' unsloth/models/loader.py
* '''Lines:''' L120-620

=== Signature ===
<syntaxhighlight lang="python">
class FastLanguageModel:
    @staticmethod
    def from_pretrained(
        model_name: str = "unsloth/Llama-3.2-1B-Instruct",
        max_seq_length: int = 2048,
        dtype: Optional[torch.dtype] = None,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        load_in_16bit: bool = False,
        full_finetuning: bool = False,
        token: Optional[str] = None,
        device_map: str = "sequential",
        rope_scaling: Optional[dict] = None,
        fix_tokenizer: bool = True,
        trust_remote_code: bool = False,
        use_gradient_checkpointing: str = "unsloth",
        resize_model_vocab: Optional[int] = None,
        revision: Optional[str] = None,
        use_exact_model_name: bool = False,
        offload_embedding: bool = False,
        float32_mixed_precision: Optional[bool] = None,
        *args,
        **kwargs,
    ) -> Tuple[PeftModel, PreTrainedTokenizer]:
        """
        Load a language model with Unsloth optimizations.

        Args:
            model_name: HuggingFace model ID or local path
            max_seq_length: Maximum sequence length for RoPE scaling
            dtype: Model dtype (None=auto-detect, torch.float16, torch.bfloat16)
            load_in_4bit: Use 4-bit NF4 quantization (QLoRA)
            load_in_8bit: Use 8-bit quantization
            load_in_16bit: Use 16-bit precision
            full_finetuning: Disable LoRA, train all parameters
            token: HuggingFace API token for private models
            device_map: Device placement strategy
            use_gradient_checkpointing: "unsloth" for optimized checkpointing

        Returns:
            Tuple of (model, tokenizer) ready for fine-tuning
        """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model_name || str || Yes || HuggingFace model ID (e.g., "unsloth/Llama-3.2-1B-Instruct") or local path
|-
| max_seq_length || int || No (default: 2048) || Maximum sequence length, affects RoPE scaling
|-
| dtype || torch.dtype || No (default: auto) || Model precision (float16, bfloat16, or None for auto)
|-
| load_in_4bit || bool || No (default: True) || Enable 4-bit NF4 quantization for QLoRA
|-
| token || str || No || HuggingFace token for accessing gated/private models
|-
| use_gradient_checkpointing || str || No (default: "unsloth") || "unsloth" for optimized, "True" for standard
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| model || PeftModel || Quantized model ready for LoRA adapter injection
|-
| tokenizer || PreTrainedTokenizer || Associated tokenizer with padding configured
|}

== Usage Examples ==

=== Basic QLoRA Loading ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# Load model with 4-bit quantization (default)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct",
    max_seq_length = 2048,
    load_in_4bit = True,
    dtype = None,  # Auto-detect (bfloat16 on Ampere+, float16 otherwise)
)

# Model is now ready for get_peft_model()
print(f"Model dtype: {model.dtype}")
print(f"Tokenizer vocab size: {len(tokenizer)}")
</syntaxhighlight>

=== Extended Context Length ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# Load with extended context (RoPE scaling applied automatically)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B-Instruct",
    max_seq_length = 8192,  # Extended from default 2048
    load_in_4bit = True,
)
</syntaxhighlight>

=== Loading Private/Gated Models ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# Access gated models with HF token
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "meta-llama/Llama-3.2-3B-Instruct",  # Gated model
    max_seq_length = 2048,
    load_in_4bit = True,
    token = "hf_your_token_here",  # Required for gated models
)
</syntaxhighlight>

=== Pre-Quantized Models (Faster Loading) ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# Unsloth provides pre-quantized models for faster loading
# These skip the quantization step entirely
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",  # Pre-quantized
    max_seq_length = 2048,
    load_in_4bit = True,  # Already quantized, just loads
)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:unslothai_unsloth_Model_Loading]]

=== Requires Environment ===
* [[requires_env::Environment:unslothai_unsloth_CUDA]]

