{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth GitHub|https://github.com/unslothai/unsloth]]
* [[source::Doc|Unsloth Documentation|https://docs.unsloth.ai/]]
|-
! Domains
| [[domain::LLMs]], [[domain::Model_Loading]], [[domain::Fine_Tuning]]
|-
! Last Updated
| [[last_updated::2025-12-12 00:00 GMT]]
|}

== Overview ==
Concrete tool for loading and patching language models with Unsloth's 2x faster optimizations provided by the Unsloth library.

=== Description ===
`FastLanguageModel` is the primary entry point for using Unsloth. It provides a unified API to load any supported language model (Llama, Mistral, Qwen, Gemma, etc.) with automatic patching for 2x faster training and 70% less VRAM. The class handles model loading, quantization (4-bit, 8-bit, 16-bit), and optimization kernel injection transparently.

=== Usage ===
Import this class when you need to load a language model for fine-tuning with Unsloth optimizations. Use it as the first step in any QLoRA, SFT, DPO, or GRPO workflow. Preferred over direct HuggingFace model loading when speed and memory efficiency are priorities.

== Code Signature ==
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

class FastLanguageModel:
    @staticmethod
    def from_pretrained(
        model_name: str,
        max_seq_length: int = 2048,
        dtype: Optional[torch.dtype] = None,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        full_finetuning: bool = False,
        token: Optional[str] = None,
        device_map: str = "auto",
        trust_remote_code: bool = False,
        **kwargs
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load model with Unsloth optimizations."""
        ...
    
    @staticmethod
    def get_peft_model(
        model: PreTrainedModel,
        r: int = 16,
        target_modules: List[str] = ["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
        lora_alpha: int = 16,
        lora_dropout: float = 0,
        bias: str = "none",
        use_gradient_checkpointing: Union[bool, str] = "unsloth",
        random_state: int = 3407,
        max_seq_length: int = 2048,
        use_rslora: bool = False,
        **kwargs
    ) -> PeftModel:
        """Add LoRA adapters with Unsloth optimizations."""
        ...
</syntaxhighlight>

== I/O Contract ==
* **Consumes:**
    * `model_name`: String path to HuggingFace model or local path
    * `max_seq_length`: Integer, maximum context length
    * `token`: Optional HuggingFace token for gated models
* **Produces:**
    * `model`: Patched `PreTrainedModel` with Unsloth optimizations
    * `tokenizer`: Corresponding `PreTrainedTokenizer`

== Example Usage ==
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# Load model with 4-bit quantization
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = 2048,
    load_in_4bit = True,
    # token = "hf_...",  # For gated models
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
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

