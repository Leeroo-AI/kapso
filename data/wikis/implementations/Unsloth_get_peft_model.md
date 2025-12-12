{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth GitHub|https://github.com/unslothai/unsloth]]
* [[source::Doc|Unsloth Documentation|https://docs.unsloth.ai/]]
* [[source::Paper|LoRA Paper|https://arxiv.org/abs/2106.09685]]
|-
! Domains
| [[domain::Fine_Tuning]], [[domain::PEFT]], [[domain::LLMs]]
|-
! Last Updated
| [[last_updated::2025-12-12 00:00 GMT]]
|}

== Overview ==
Concrete tool for adding LoRA adapters to models with Unsloth's memory-optimized implementation.

=== Description ===
`get_peft_model` is the function (accessed via `FastLanguageModel.get_peft_model` or `FastModel.get_peft_model`) that injects LoRA adapters into a loaded model. It applies Unsloth's custom optimizations including the "unsloth" gradient checkpointing mode that provides 30% additional VRAM savings. This is where LoRA hyperparameters (rank, alpha, target modules) are configured.

=== Usage ===
Call this function immediately after loading a model with `FastLanguageModel.from_pretrained` or `FastModel.from_pretrained`. Required step before training - it transforms the frozen model into a parameter-efficient trainable model.

== Code Signature ==
<syntaxhighlight lang="python">
# Accessed as a static method
FastLanguageModel.get_peft_model(
    model: PreTrainedModel,
    r: int = 16,
    target_modules: List[str] = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_alpha: int = 16,
    lora_dropout: float = 0,
    bias: str = "none",
    use_gradient_checkpointing: Union[bool, str] = "unsloth",
    random_state: int = 3407,
    max_seq_length: int = 2048,
    use_rslora: bool = False,
    loftq_config: Optional[dict] = None,
    **kwargs
) -> PeftModel
</syntaxhighlight>

== I/O Contract ==
* **Consumes:**
    * `model`: Pre-loaded model from `from_pretrained`
    * `r`: LoRA rank (typically 8, 16, or 32)
    * `target_modules`: List of layer names to apply LoRA
    * `lora_alpha`: LoRA scaling factor
    * `use_gradient_checkpointing`: Memory optimization setting
* **Produces:**
    * `PeftModel` with LoRA adapters and Unsloth optimizations

== Example Usage ==
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# Load model first
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = 2048,
    load_in_4bit = True,
)

# Add LoRA adapters with Unsloth optimizations
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,                          # LoRA rank
    target_modules = [               # Layers to adapt
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = 16,                 # Scaling factor
    lora_dropout = 0,                # 0 is optimized
    bias = "none",                   # "none" is optimized
    use_gradient_checkpointing = "unsloth",  # 30% extra VRAM savings
    random_state = 3407,
)

# Check trainable parameters
model.print_trainable_parameters()
# Output: trainable params: 41,943,040 || all params: 8,030,261,248 || trainable%: 0.5222
</syntaxhighlight>

== Related Pages ==
=== Context & Requirements ===
* [[requires_env::Environment:Unsloth_CUDA_Environment]]
* [[requires_env::Environment:Unsloth_Colab_Environment]]

=== Tips and Tricks ===
* [[uses_heuristic::Heuristic:Unsloth_Gradient_Checkpointing_Optimization]]
* [[uses_heuristic::Heuristic:QLoRA_Target_Modules_Selection]]
* [[uses_heuristic::Heuristic:LoRA_Rank_Selection]]

