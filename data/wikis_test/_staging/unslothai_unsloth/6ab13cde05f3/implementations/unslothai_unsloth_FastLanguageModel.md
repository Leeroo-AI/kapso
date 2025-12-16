{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|Unsloth Documentation|https://docs.unsloth.ai]]
* [[source::Paper|QLoRA|https://arxiv.org/abs/2305.14314]]
|-
! Domains
| [[domain::NLP]], [[domain::Fine_Tuning]], [[domain::Model_Loading]], [[domain::Quantization]]
|-
! Last Updated
| [[last_updated::2025-12-16 18:00 GMT]]
|}

== Overview ==

Concrete tool for loading and optimizing Large Language Models with 4-bit/8-bit quantization, provided by the Unsloth library.

=== Description ===

`FastLanguageModel` is the primary entry point for loading pre-trained language models in Unsloth. It extends the base `FastLlamaModel` class and provides a unified API for:

* **Multi-architecture support**: Automatically detects and dispatches to architecture-specific optimizations (Llama, Mistral, Gemma, Qwen, Cohere, Granite, Falcon)
* **Quantization handling**: Supports 4-bit (QLoRA via bitsandbytes), 8-bit, 16-bit, and FP8 quantization modes
* **PEFT integration**: Seamlessly loads existing LoRA adapters from HuggingFace Hub
* **vLLM acceleration**: Optional `fast_inference=True` for vLLM-backed generation during RL training
* **Automatic optimization**: Patches attention backends, enables gradient checkpointing, and applies Unsloth's custom kernels

The class handles the complexity of model loading including token authentication, model name mapping to pre-quantized versions, and automatic dtype selection based on hardware capabilities.

=== Usage ===

Import `FastLanguageModel` when you need to load a language model for fine-tuning with Unsloth's optimizations. This is the first step in any QLoRA, LoRA, or full fine-tuning workflow.

Use cases:
* Loading a model for QLoRA fine-tuning with 4-bit quantization
* Loading pre-existing LoRA adapters for continued training
* Loading models with vLLM backend for reinforcement learning

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unslothai_unsloth]
* '''File:''' [https://github.com/unslothai/unsloth/blob/main/unsloth/models/loader.py unsloth/models/loader.py]
* '''Lines:''' 120-620 (FastLanguageModel.from_pretrained)

=== Signature ===
<syntaxhighlight lang="python">
class FastLanguageModel(FastLlamaModel):
    @staticmethod
    def from_pretrained(
        model_name: str = "unsloth/Llama-3.2-1B-Instruct",
        max_seq_length: int = 2048,
        dtype = None,  # Auto-detects: bfloat16 if supported, else float16
        load_in_4bit: bool = True,   # 4-bit QLoRA quantization
        load_in_8bit: bool = False,  # 8-bit LoRA quantization
        load_in_16bit: bool = False, # 16-bit LoRA (no quantization)
        full_finetuning: bool = False,
        token: str = None,           # HuggingFace token for gated models
        device_map: str = "sequential",
        rope_scaling = None,
        fix_tokenizer: bool = True,
        trust_remote_code: bool = False,
        use_gradient_checkpointing: str = "unsloth",  # 30% VRAM reduction
        resize_model_vocab: int = None,
        revision: str = None,
        use_exact_model_name: bool = False,
        offload_embedding: bool = False,
        float32_mixed_precision: bool = None,
        fast_inference: bool = False,  # Enable vLLM backend
        gpu_memory_utilization: float = 0.5,
        float8_kv_cache: bool = False,
        random_state: int = 3407,
        max_lora_rank: int = 64,
        disable_log_stats: bool = True,
        qat_scheme: str = None,
        load_in_fp8: bool = False,
        unsloth_tiled_mlp: bool = False,
        *args,
        **kwargs,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load a pre-trained language model with Unsloth optimizations.

        Returns:
            Tuple of (model, tokenizer) ready for fine-tuning.
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
| model_name || str || Yes || HuggingFace model ID or local path (e.g., "unsloth/Llama-3.2-1B-Instruct")
|-
| max_seq_length || int || No || Maximum sequence length for RoPE scaling (default: 2048)
|-
| dtype || torch.dtype || No || Compute dtype; auto-detected if None (bfloat16 preferred)
|-
| load_in_4bit || bool || No || Enable 4-bit quantization via bitsandbytes (default: True)
|-
| load_in_8bit || bool || No || Enable 8-bit quantization (default: False)
|-
| token || str || No || HuggingFace token for gated models
|-
| fast_inference || bool || No || Enable vLLM backend for fast generation (default: False)
|-
| use_gradient_checkpointing || str || No || "unsloth" for optimized checkpointing, True/False for standard
|-
| gpu_memory_utilization || float || No || vLLM GPU memory fraction (default: 0.5)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| model || PreTrainedModel || Optimized model with Unsloth patches applied
|-
| tokenizer || PreTrainedTokenizer || Corresponding tokenizer with fixes applied
|}

== Usage Examples ==

=== Basic QLoRA Loading ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel
import torch

# Load model with 4-bit quantization (QLoRA)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,   # Enable 4-bit quantization
    dtype=None,          # Auto-detect best dtype
)

# Model is now ready for get_peft_model() and training
print(f"Model loaded with {model.dtype} precision")
</syntaxhighlight>

=== Loading with vLLM for RL Training ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# Load with vLLM for fast inference during GRPO/PPO
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-8B-bnb-4bit",
    max_seq_length=4096,
    load_in_4bit=True,
    fast_inference=True,        # Enable vLLM backend
    gpu_memory_utilization=0.6, # Reserve GPU memory for gradients
    max_lora_rank=64,           # Support higher LoRA ranks for RL
)
</syntaxhighlight>

=== Loading Existing LoRA Adapters ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# Load a previously trained LoRA adapter
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="your-username/my-lora-adapter",  # LoRA adapter repo
    max_seq_length=2048,
    load_in_4bit=True,
    token="hf_...",  # Required for private repos
)

# Model loads with LoRA adapters already attached
</syntaxhighlight>

=== Loading Gated Models ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# Load a gated model like Llama-3
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Llama-3.2-1B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
    token="hf_...",  # Required for gated models
    trust_remote_code=False,
)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:unslothai_unsloth_Model_Loading]]

=== Requires Environment ===
* [[requires_env::Environment:unslothai_unsloth_CUDA]]

=== Uses Heuristics ===
* [[uses_heuristic::Heuristic:unslothai_unsloth_Memory_Optimization]]
* [[uses_heuristic::Heuristic:unslothai_unsloth_Mixed_Precision_Training]]
