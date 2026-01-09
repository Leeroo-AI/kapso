# Implementation: FastLanguageModel_from_pretrained

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
| [[last_updated::2026-01-09 16:00 GMT]]
|}

== Overview ==

Concrete tool for loading Large Language Models with 4-bit quantization for memory-efficient QLoRA fine-tuning, provided by the Unsloth library.

=== Description ===

`FastLanguageModel.from_pretrained` is the primary entry point for loading pre-trained language models with Unsloth's optimizations. It handles automatic 4-bit NF4 quantization via bitsandbytes, device mapping, attention backend selection (Flash Attention 2, xformers, or SDPA), and model-specific patches. The function returns both the model and tokenizer in a single call, ready for LoRA adapter injection.

Key capabilities for QLoRA fine-tuning:
* **4-bit NF4 quantization** - Reduces memory footprint by ~75% while maintaining accuracy
* **Automatic model detection** - Routes to appropriate model class (LLaMA, Mistral, Qwen, Gemma, etc.)
* **Unsloth gradient checkpointing** - Memory-efficient alternative to standard gradient checkpointing
* **Tokenizer patching** - Fixes common tokenizer issues (padding, special tokens)

=== Usage ===

Use this function when starting a QLoRA fine-tuning workflow. Import and call at the beginning of your training script to load a base model for fine-tuning. This is the first step in the standard QLoRA training pipeline before applying LoRA adapters with `get_peft_model`.

NOT for:
* Reinforcement learning workflows (use `fast_inference=True` variant instead)
* Full fine-tuning (set `full_finetuning=True` which routes to FastModel)
* Vision-language models (use `FastVisionModel.from_pretrained`)

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unsloth]
* '''File:''' unsloth/models/loader.py
* '''Lines:''' L121-700

=== Signature ===
<syntaxhighlight lang="python">
class FastLanguageModel(FastLlamaModel):
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
        fast_inference: bool = False,
        gpu_memory_utilization: float = 0.5,
        float8_kv_cache: bool = False,
        random_state: int = 3407,
        max_lora_rank: int = 64,
        disable_log_stats: bool = True,
        qat_scheme: Optional[str] = None,
        load_in_fp8: bool = False,
        unsloth_tiled_mlp: bool = False,
        **kwargs,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load a pretrained language model with Unsloth optimizations.

        Args:
            model_name: HuggingFace model ID or local path
            max_seq_length: Maximum sequence length for training
            dtype: Compute dtype (None for auto-detection)
            load_in_4bit: Enable 4-bit NF4 quantization for QLoRA
            load_in_8bit: Enable 8-bit quantization
            load_in_16bit: Load in float16 without quantization
            full_finetuning: Enable full parameter training (no LoRA)
            token: HuggingFace token for private models
            device_map: Device placement strategy
            use_gradient_checkpointing: "unsloth" for optimized checkpointing
            fix_tokenizer: Apply tokenizer fixes
            trust_remote_code: Allow custom model code

        Returns:
            Tuple of (model, tokenizer) with optimizations applied
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
| max_seq_length || int || No (default: 2048) || Maximum sequence length for RoPE scaling and attention
|-
| load_in_4bit || bool || No (default: True) || Enable 4-bit NF4 quantization via bitsandbytes
|-
| dtype || torch.dtype || No (default: None) || Compute dtype; None for auto-detection based on GPU
|-
| token || str || No || HuggingFace Hub token for private/gated models
|-
| use_gradient_checkpointing || str || No (default: "unsloth") || Gradient checkpointing mode
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| model || PreTrainedModel || Quantized model with Unsloth patches applied
|-
| tokenizer || PreTrainedTokenizer || Tokenizer with padding and special token fixes
|}

== Usage Examples ==

=== Basic QLoRA Loading ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# Load model with 4-bit quantization for QLoRA fine-tuning
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct",
    max_seq_length = 2048,
    load_in_4bit = True,   # Enable QLoRA quantization
    dtype = None,          # Auto-detect (float16 or bfloat16)
)

# Model is now ready for LoRA adapter injection
print(f"Model dtype: {model.dtype}")
print(f"Tokenizer vocab size: {len(tokenizer)}")
</syntaxhighlight>

=== Loading with Custom Sequence Length ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# Load for longer context fine-tuning
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.1-8B-Instruct",
    max_seq_length = 8192,  # Extended context
    load_in_4bit = True,
    dtype = None,
    use_gradient_checkpointing = "unsloth",  # Memory-efficient
)
</syntaxhighlight>

=== Loading Private/Gated Models ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# Load gated model with HuggingFace token
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "meta-llama/Llama-3.2-1B-Instruct",
    max_seq_length = 2048,
    load_in_4bit = True,
    token = "hf_your_token_here",  # Required for gated models
)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Unslothai_Unsloth_Model_Loading]]

=== Requires Environment ===
* [[requires_env::Environment:Unslothai_Unsloth_CUDA_GPU_Environment]]

=== Uses Heuristics ===
* [[uses_heuristic::Heuristic:Unslothai_Unsloth_Gradient_Checkpointing_Tip]]
* [[uses_heuristic::Heuristic:Unslothai_Unsloth_BFloat16_vs_Float16_Tip]]
