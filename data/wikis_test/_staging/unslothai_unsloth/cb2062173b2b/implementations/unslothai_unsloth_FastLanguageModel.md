{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|Unsloth Docs|https://docs.unsloth.ai]]
* [[source::Doc|Fine-tuning Guide|https://docs.unsloth.ai/get-started/fine-tuning-guide]]
|-
! Domains
| [[domain::LLMs]], [[domain::Model_Loading]], [[domain::Quantization]], [[domain::PEFT]]
|-
! Last Updated
| [[last_updated::2025-12-16 14:00 GMT]]
|}

== Overview ==
Concrete tool for loading and optimizing pre-trained language models with 4-bit/8-bit/16-bit quantization provided by the Unsloth library.

=== Description ===
`FastLanguageModel` is the primary user-facing API for loading transformer-based language models in Unsloth. It serves as a unified entry point that:

1. **Auto-detects model architecture** (Llama, Mistral, Qwen, Gemma, etc.) and routes to the appropriate optimized implementation
2. **Applies quantization** via BitsAndBytes (4-bit NF4, 8-bit) or FP8 for memory efficiency
3. **Patches the model** with Unsloth's optimized Triton kernels for attention, normalization, and activation functions
4. **Handles PEFT adapters** automatically loading LoRA checkpoints when detected
5. **Integrates with vLLM** for fast inference when `fast_inference=True`

The class extends `FastLlamaModel` and delegates to architecture-specific implementations (`FastMistralModel`, `FastQwen2Model`, `FastGemmaModel`, etc.) based on the detected model type. For unsupported architectures, it falls back to `FastModel` which provides generic optimizations.

=== Usage ===
Import this class when you need to:
- Load a pre-trained LLM for fine-tuning with QLoRA/LoRA
- Load a 4-bit quantized model to reduce VRAM usage
- Resume training from a saved LoRA checkpoint
- Set up fast inference with vLLM integration

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unslothai_unsloth]
* '''File:''' [https://github.com/unslothai/unsloth/blob/main/unsloth/models/loader.py#L120-L620 unsloth/models/loader.py]
* '''Lines:''' 120-620

Source Files: unsloth/models/loader.py:L120-L620

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
        load_in_fp8: Union[bool, str] = False,
        unsloth_tiled_mlp: bool = False,
        *args,
        **kwargs,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load and optimize a language model for fine-tuning or inference.

        Args:
            model_name: HuggingFace model name or local path
            max_seq_length: Maximum sequence length (supports RoPE scaling)
            dtype: Compute dtype (auto-detected if None)
            load_in_4bit: Enable 4-bit quantization (QLoRA)
            load_in_8bit: Enable 8-bit quantization
            load_in_16bit: Enable 16-bit LoRA
            full_finetuning: Disable LoRA for full fine-tuning
            token: HuggingFace token for private models
            device_map: Device placement strategy
            use_gradient_checkpointing: "unsloth" for 30% less VRAM
            fast_inference: Enable vLLM fast inference
            gpu_memory_utilization: vLLM memory fraction (0.0-1.0)
            load_in_fp8: FP8 quantization (True, False, or "block")

        Returns:
            Tuple of (model, tokenizer)
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
| model_name || str || No (default: "unsloth/Llama-3.2-1B-Instruct") || HuggingFace model ID or local path
|-
| max_seq_length || int || No (default: 2048) || Maximum sequence length for training
|-
| dtype || torch.dtype || No (auto) || Compute precision (float16, bfloat16, float32)
|-
| load_in_4bit || bool || No (default: True) || Enable 4-bit NF4 quantization
|-
| load_in_8bit || bool || No (default: False) || Enable 8-bit quantization
|-
| token || str || No || HuggingFace API token for private models
|-
| use_gradient_checkpointing || str || No (default: "unsloth") || Gradient checkpointing mode
|-
| fast_inference || bool || No (default: False) || Enable vLLM acceleration
|-
| gpu_memory_utilization || float || No (default: 0.5) || vLLM GPU memory fraction
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| model || PreTrainedModel || Optimized model with Unsloth patches applied
|-
| tokenizer || PreTrainedTokenizer || Tokenizer with optional fixes applied
|}

== Usage Examples ==

=== Basic 4-bit Loading ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel
import torch

# Load model with 4-bit quantization (default)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
    dtype=None,  # Auto-detect (bfloat16 for Ampere+)
)

# Model is ready for LoRA fine-tuning
print(f"Model loaded: {model.config.model_type}")
</syntaxhighlight>

=== Loading with vLLM Fast Inference ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# Load for GRPO/RL training with fast generation
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    max_seq_length=2048,
    load_in_4bit=False,  # 16-bit for LoRA with vLLM
    fast_inference=True,  # Enable vLLM
    gpu_memory_utilization=0.8,
    max_lora_rank=64,
)

# Generate with vLLM speed
inputs = tokenizer("Hello, ", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
</syntaxhighlight>

=== Loading a LoRA Checkpoint ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# Load previously saved LoRA adapter
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./my_lora_checkpoint",  # Local path with adapter_config.json
    max_seq_length=2048,
    load_in_4bit=True,
)

# Model has LoRA adapters applied and is ready for inference or continued training
</syntaxhighlight>

=== Full Fine-tuning (No LoRA) ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# Full fine-tuning requires more VRAM but updates all weights
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Llama-3.2-1B",
    max_seq_length=2048,
    full_finetuning=True,  # Disables LoRA
    load_in_4bit=False,
    load_in_8bit=False,
)
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:unslothai_unsloth_CUDA_Compute]]
* [[uses_heuristic::Heuristic:unslothai_unsloth_Memory_Management]]
* [[uses_heuristic::Heuristic:unslothai_unsloth_Gradient_Checkpointing]]
