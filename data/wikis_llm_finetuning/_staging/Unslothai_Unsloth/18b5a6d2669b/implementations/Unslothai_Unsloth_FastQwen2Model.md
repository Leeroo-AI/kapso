# Implementation: FastQwen2Model

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|Qwen|https://qwenlm.github.io/]]
|-
! Domains
| [[domain::Models]], [[domain::NLP]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2026-01-09 15:46 GMT]]
|}

== Overview ==
Optimized patching class for Qwen 2 models reusing LLaMA optimizations.

=== Description ===
`FastQwen2Model` provides Unsloth optimizations for Qwen 2 models by inheriting from `FastLlamaModel` and applying Qwen2-specific patches. Since Qwen 2 shares architectural similarities with LLaMA, it reuses the LLaMA attention and decoder layer fast forward implementations.

=== Usage ===
Used through `FastLanguageModel.from_pretrained()` when loading Qwen 2 models. Can also be used directly for more control.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth Unslothai_Unsloth]
* '''File:''' [https://github.com/unslothai/unsloth/blob/main/unsloth/models/qwen2.py unsloth/models/qwen2.py]
* '''Lines:''' 1-101

=== Key Methods ===
<syntaxhighlight lang="python">
class FastQwen2Model(FastLlamaModel):
    @staticmethod
    def pre_patch():
        """
        Apply Qwen2-specific patches.

        Patches applied:
        - Qwen2Attention.forward -> LlamaAttention_fast_forward
        - Qwen2DecoderLayer.forward -> LlamaDecoderLayer_fast_forward
        - Qwen2Model.forward -> LlamaModel_fast_forward
        - Qwen2ForCausalLM.forward -> CausalLM_fast_forward
        - Qwen2RotaryEmbedding -> LlamaRotaryEmbedding
        """

    @staticmethod
    def from_pretrained(
        model_name: str = "Qwen/Qwen2-7B",
        max_seq_length: int = 4096,
        dtype = None,
        load_in_4bit: bool = True,
        token: str = None,
        device_map: str = "sequential",
        rope_scaling = None,  # Qwen2 does not support RoPE scaling
        fix_tokenizer: bool = True,
        **kwargs,
    ):
        """Load Qwen2 model with Unsloth optimizations."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from unsloth.models.qwen2 import FastQwen2Model

# Or through the unified API
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2-7B",
    max_seq_length=4096,
    load_in_4bit=True,
)
</syntaxhighlight>

== I/O Contract ==

=== from_pretrained Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model_name || str || No || Model ID (default: Qwen/Qwen2-7B)
|-
| max_seq_length || int || No || Max sequence length (default: 4096)
|-
| load_in_4bit || bool || No || Enable 4-bit quantization (default: True)
|-
| dtype || dtype || No || Data type (auto-detected if None)
|-
| rope_scaling || None || No || Not supported for Qwen2
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| model || PreTrainedModel || Qwen2 model with patches applied
|-
| tokenizer || PreTrainedTokenizer || Qwen2 tokenizer
|}

== Usage Examples ==

=== Basic Loading ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2-7B-Instruct",
    max_seq_length=4096,
    dtype=None,
    load_in_4bit=True,
)

# Add LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
)
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:Unslothai_Unsloth_CUDA_GPU_Environment]]
