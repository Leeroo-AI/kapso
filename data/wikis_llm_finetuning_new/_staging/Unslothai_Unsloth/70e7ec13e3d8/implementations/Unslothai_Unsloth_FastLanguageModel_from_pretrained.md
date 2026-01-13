# Implementation: FastLanguageModel_from_pretrained

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|Unsloth Documentation|https://docs.unsloth.ai]]
* [[source::Paper|QLoRA|https://arxiv.org/abs/2305.14314]]
|-
! Domains
| [[domain::NLP]], [[domain::Model_Loading]], [[domain::Quantization]]
|-
! Last Updated
| [[last_updated::2026-01-12 00:00 GMT]]
|}

== Overview ==

Concrete tool for loading pre-trained language models with 4-bit quantization provided by the Unsloth library.

=== Description ===

`FastLanguageModel.from_pretrained` is the primary entry point for loading language models in Unsloth. It wraps HuggingFace's `AutoModelForCausalLM` with additional optimizations including:

* Automatic 4-bit NF4 quantization via bitsandbytes
* Model architecture detection and optimized kernel patching
* RoPE (Rotary Position Embedding) extension for longer context lengths
* Tokenizer configuration and special token handling
* Optional vLLM integration for fast inference during RL training

The function dispatches to architecture-specific loaders (FastLlamaModel, FastMistralModel, FastQwen2Model, etc.) based on the model's config.

=== Usage ===

Import this function when you need to load a language model for QLoRA fine-tuning. Use it at the start of any Unsloth training workflow. For reinforcement learning workflows requiring fast inference, set `fast_inference=True` to enable vLLM backend.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unsloth]
* '''File:''' unsloth/models/loader.py
* '''Lines:''' 121-676

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
        use_gradient_checkpointing: Union[str, bool] = "unsloth",
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
        load_in_fp8: Union[bool, str] = False,
        unsloth_tiled_mlp: bool = False,
        *args,
        **kwargs,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load a pre-trained language model with optional quantization.

        Args:
            model_name: HuggingFace model ID or local path
            max_seq_length: Maximum sequence length (can exceed model default)
            dtype: Model precision (auto-detected if None)
            load_in_4bit: Enable 4-bit QLoRA quantization
            load_in_8bit: Enable 8-bit LoRA quantization
            load_in_16bit: Enable 16-bit LoRA (no quantization)
            full_finetuning: Enable full model training (no LoRA)
            token: HuggingFace API token for private models
            device_map: Device placement strategy
            rope_scaling: Custom RoPE scaling configuration
            fix_tokenizer: Auto-fix tokenizer issues
            trust_remote_code: Allow remote code execution
            use_gradient_checkpointing: Memory optimization ("unsloth" recommended)
            fast_inference: Enable vLLM backend for GRPO/RL workflows
            gpu_memory_utilization: vLLM GPU memory fraction (0.0-1.0)
            max_lora_rank: Maximum LoRA rank for vLLM adapter loading

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
| max_seq_length || int || No || Maximum sequence length for training (default: 2048)
|-
| dtype || torch.dtype || No || Model precision (auto-detected from model config if None)
|-
| load_in_4bit || bool || No || Enable 4-bit NF4 quantization (default: True)
|-
| token || str || No || HuggingFace API token for private/gated models
|-
| device_map || str || No || Device placement strategy (default: "sequential")
|-
| fast_inference || bool || No || Enable vLLM backend for RL workflows (default: False)
|-
| use_gradient_checkpointing || str/bool || No || Gradient checkpointing mode (default: "unsloth")
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| model || PreTrainedModel || Quantized model with Unsloth optimizations applied
|-
| tokenizer || PreTrainedTokenizer || Configured tokenizer with proper special tokens
|}

== Usage Examples ==

=== Basic QLoRA Loading ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# Load a 4-bit quantized model for QLoRA training
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=2048,
    dtype=None,  # Auto-detect
    load_in_4bit=True,
)

# Model is now ready for LoRA adapter injection
print(f"Model loaded with {model.num_parameters():,} parameters")
</syntaxhighlight>

=== Loading with Extended Context ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# Load model with extended context length
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    max_seq_length=8192,  # Extended from default 4096
    dtype=None,
    load_in_4bit=True,
    token="hf_xxx",  # Required for gated models
)
</syntaxhighlight>

=== Loading for GRPO/RL Training ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# Load with vLLM backend for fast inference during RL
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
    fast_inference=True,  # Enable vLLM
    gpu_memory_utilization=0.6,
    max_lora_rank=64,
)

# model.vllm_engine is now available for generation
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Unslothai_Unsloth_Model_Loading]]

=== Requires Environment ===
* [[requires_env::Environment:Unslothai_Unsloth_CUDA_11]]
