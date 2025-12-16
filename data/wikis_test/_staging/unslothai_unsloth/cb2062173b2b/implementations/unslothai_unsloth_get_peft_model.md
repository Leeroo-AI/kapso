{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|Unsloth Docs|https://docs.unsloth.ai]]
* [[source::Paper|LoRA|https://arxiv.org/abs/2106.09685]]
* [[source::Paper|QLoRA|https://arxiv.org/abs/2305.14314]]
|-
! Domains
| [[domain::LLMs]], [[domain::PEFT]], [[domain::LoRA]], [[domain::Fine_Tuning]]
|-
! Last Updated
| [[last_updated::2025-12-16 14:00 GMT]]
|}

== Overview ==
Concrete tool for applying LoRA (Low-Rank Adaptation) adapters to a pre-trained model provided by the Unsloth library.

=== Description ===
`FastLanguageModel.get_peft_model()` is a static method that applies LoRA adapters to a loaded model, enabling parameter-efficient fine-tuning. This method:

1. **Configures LoRA parameters** including rank, alpha, dropout, and target modules
2. **Applies PEFT adapters** to specified attention and MLP layers
3. **Enables Unsloth gradient checkpointing** for 30% VRAM reduction
4. **Validates configuration** ensuring rank > 0 and no duplicate adapter application
5. **Supports vLLM integration** for fast inference during RL training

The method wraps the PEFT library's `get_peft_model()` with Unsloth-specific optimizations and validations. It targets specific projection layers (q_proj, k_proj, v_proj, o_proj for attention; gate_proj, up_proj, down_proj for MLP) while maintaining compatibility with the base model's architecture.

=== Usage ===
Import and use this method when you need to:
- Add LoRA adapters to a loaded model for fine-tuning
- Configure which layers to adapt (attention, MLP, or both)
- Set up efficient gradient checkpointing
- Prepare a model for QLoRA training

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unslothai_unsloth]
* '''File:''' [https://github.com/unslothai/unsloth/blob/main/unsloth/models/llama.py#L2578-L2800 unsloth/models/llama.py]
* '''Lines:''' 2578-2800

Source Files: unsloth/models/llama.py:L2578-L2800; unsloth/models/vision.py:L910-L1100

=== Signature ===
<syntaxhighlight lang="python">
class FastLanguageModel:
    @staticmethod
    def get_peft_model(
        model: PreTrainedModel,
        r: int = 16,
        target_modules: List[str] = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        bias: str = "none",
        layers_to_transform: Optional[List[int]] = None,
        layers_pattern: Optional[str] = None,
        use_gradient_checkpointing: str = "unsloth",
        random_state: int = 3407,
        max_seq_length: int = 2048,
        use_rslora: bool = False,
        modules_to_save: Optional[List[str]] = None,
        init_lora_weights: bool = True,
        loftq_config: dict = {},
        temporary_location: str = "_unsloth_temporary_saved_buffers",
        qat_scheme: Optional[str] = None,
        **kwargs,
    ) -> PeftModel:
        """
        Apply LoRA adapters to a model for parameter-efficient fine-tuning.

        Args:
            model: The base model to adapt
            r: LoRA rank (8, 16, 32, 64, 128 are common)
            target_modules: List of module names to apply LoRA to
            lora_alpha: LoRA scaling factor (typically equals r)
            lora_dropout: Dropout probability (0 is optimized)
            bias: Bias training mode ("none", "all", "lora_only")
            use_gradient_checkpointing: "unsloth" for 30% less VRAM
            random_state: Random seed for reproducibility
            use_rslora: Enable Rank-Stabilized LoRA
            modules_to_save: Additional modules to save unmodified
            qat_scheme: Quantization-aware training scheme

        Returns:
            PeftModel with LoRA adapters applied
        """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel
# Used as: FastLanguageModel.get_peft_model(model, ...)
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model || PreTrainedModel || Yes || Base model from FastLanguageModel.from_pretrained()
|-
| r || int || No (default: 16) || LoRA rank - higher = more capacity, more memory
|-
| target_modules || List[str] || No || Layers to apply LoRA (attention + MLP projections)
|-
| lora_alpha || int || No (default: 16) || Scaling factor (effective scale = alpha/r)
|-
| lora_dropout || float || No (default: 0.0) || Dropout on LoRA layers (0 is optimized)
|-
| bias || str || No (default: "none") || Whether to train biases ("none" is optimized)
|-
| use_gradient_checkpointing || str || No (default: "unsloth") || Checkpointing strategy
|-
| random_state || int || No (default: 3407) || Random seed
|-
| use_rslora || bool || No (default: False) || Enable Rank-Stabilized LoRA
|-
| modules_to_save || List[str] || No || Modules to train fully (e.g., embed_tokens)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| model || PeftModel || Model with LoRA adapters applied and ready for training
|}

== Usage Examples ==

=== Standard LoRA Configuration ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# Load base model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Apply LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,  # 0 is optimized for Unsloth
    bias="none",  # "none" is optimized
    use_gradient_checkpointing="unsloth",  # 30% less VRAM
    random_state=3407,
)

# Model is ready for training
print(f"Trainable parameters: {model.print_trainable_parameters()}")
</syntaxhighlight>

=== Higher Rank for Reasoning Tasks ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    max_seq_length=2048,
    load_in_4bit=False,  # 16-bit for RL
    fast_inference=True,
    max_lora_rank=128,  # Allow higher ranks with vLLM
)

# Higher rank for complex reasoning tasks (GRPO, math, coding)
model = FastLanguageModel.get_peft_model(
    model,
    r=64,  # Higher rank for more capacity
    lora_alpha=64,  # Match alpha to rank
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    use_gradient_checkpointing="unsloth",
)
</syntaxhighlight>

=== Attention-Only LoRA ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/mistral-7b-bnb-4bit",
    max_seq_length=4096,
    load_in_4bit=True,
)

# Only adapt attention layers (fewer parameters, faster training)
model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Attention only
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
)
</syntaxhighlight>

=== With Rank-Stabilized LoRA (rsLoRA) ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=2048,
)

# rsLoRA scales alpha by sqrt(r) for more stable training at high ranks
model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    lora_alpha=64,
    use_rslora=True,  # Enable rank-stabilized LoRA
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:unslothai_unsloth_CUDA_Compute]]
* [[uses_heuristic::Heuristic:unslothai_unsloth_LoRA_Rank_Selection]]
