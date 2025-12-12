{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth GitHub|https://github.com/unslothai/unsloth]]
* [[source::Doc|Unsloth Documentation|https://docs.unsloth.ai/]]
|-
! Domains
| [[domain::LLMs]], [[domain::Model_Loading]], [[domain::Multi_Modal]]
|-
! Last Updated
| [[last_updated::2025-12-12 00:00 GMT]]
|}

== Overview ==
Unified model loader for all Unsloth-supported models including LLMs, Vision-Language models, and Text-to-Speech models.

=== Description ===
`FastModel` is the newer, unified API that supersedes `FastLanguageModel` for general use. It automatically detects the model type and applies appropriate optimizations. Supports language models, vision-language models (Qwen2-VL, LLaVA), and text-to-speech models. Provides the same 2x speedup and 70% VRAM reduction as `FastLanguageModel`.

=== Usage ===
Import this class as the default choice for loading any model type with Unsloth. Use `FastModel` for new projects and when working with non-language models (vision, TTS). Falls back gracefully to appropriate model-specific handling.

== Code Signature ==
<syntaxhighlight lang="python">
from unsloth import FastModel

class FastModel:
    @staticmethod
    def from_pretrained(
        model_name: str,
        max_seq_length: int = 2048,
        dtype: Optional[torch.dtype] = None,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        load_in_16bit: bool = False,
        full_finetuning: bool = False,
        token: Optional[str] = None,
        **kwargs
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load any supported model with Unsloth optimizations."""
        ...
    
    @staticmethod
    def get_peft_model(
        model: PreTrainedModel,
        r: int = 16,
        target_modules: List[str] = [...],
        lora_alpha: int = 16,
        lora_dropout: float = 0,
        bias: str = "none",
        use_gradient_checkpointing: Union[bool, str] = "unsloth",
        **kwargs
    ) -> PeftModel:
        """Add LoRA adapters with Unsloth optimizations."""
        ...
</syntaxhighlight>

== I/O Contract ==
* **Consumes:**
    * `model_name`: String path to HuggingFace model (LLM, VLM, or TTS)
    * `max_seq_length`: Integer, maximum sequence length
    * `load_in_Xbit`: Boolean flags for quantization level
* **Produces:**
    * `model`: Optimized model instance (type depends on input model)
    * `tokenizer`: Corresponding tokenizer/processor

== Example Usage ==
<syntaxhighlight lang="python">
from unsloth import FastModel

# Load any supported model (LLM, Vision, TTS)
model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-7B-bnb-4bit",  # or vision/TTS models
    max_seq_length = 2048,
    load_in_4bit = True,
)

# Add LoRA adapters
model = FastModel.get_peft_model(
    model,
    r = 16,
    lora_alpha = 16,
    use_gradient_checkpointing = "unsloth",
)
</syntaxhighlight>

== Related Pages ==
=== Context & Requirements ===
* [[requires_env::Environment:Unsloth_CUDA_Environment]]
* [[requires_env::Environment:Unsloth_Colab_Environment]]
* [[requires_env::Environment:Unsloth_Docker_Environment]]

=== Tips and Tricks ===
* [[uses_heuristic::Heuristic:Unsloth_Gradient_Checkpointing_Optimization]]
* [[uses_heuristic::Heuristic:Memory_Efficient_Attention]]

