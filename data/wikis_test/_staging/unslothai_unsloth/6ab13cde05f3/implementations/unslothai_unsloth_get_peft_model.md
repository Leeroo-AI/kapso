{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Paper|LoRA: Low-Rank Adaptation|https://arxiv.org/abs/2106.09685]]
* [[source::Paper|QLoRA|https://arxiv.org/abs/2305.14314]]
|-
! Domains
| [[domain::NLP]], [[domain::Fine_Tuning]], [[domain::LoRA]], [[domain::Parameter_Efficient_Training]]
|-
! Last Updated
| [[last_updated::2025-12-16 18:00 GMT]]
|}

== Overview ==

Concrete tool for applying Low-Rank Adaptation (LoRA) adapters to language models with fused kernel optimizations, provided by the Unsloth library.

=== Description ===

`get_peft_model` is a static method of `FastLanguageModel` (and `FastLlamaModel`) that wraps PEFT's LoRA injection with Unsloth-specific optimizations:

* **Fused LoRA operations**: Combines adapter computations with base layer operations for 2x faster training
* **Optimized dropout**: Dropout=0 enables additional kernel fusions
* **Smart gradient checkpointing**: "unsloth" mode provides 30% VRAM reduction over standard checkpointing
* **Embedding training support**: Special handling for `embed_tokens` and `lm_head` modules with mixed precision

The method intelligently handles:
- Detecting existing LoRA adapters and validating compatibility
- Offloading original embeddings to CPU when training new embeddings
- Applying LoRA to specific transformer layers via `layers_to_transform`
- Supporting RSLoRA (Rank-Stabilized LoRA) for improved training stability

=== Usage ===

Call `FastLanguageModel.get_peft_model()` after loading a model with `from_pretrained()` to add trainable LoRA adapters. This is the second step in any QLoRA/LoRA fine-tuning workflow.

Use cases:
* Adding LoRA adapters for parameter-efficient fine-tuning
* Customizing which layers receive adapters
* Training embeddings alongside LoRA adapters

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unslothai_unsloth]
* '''File:''' [https://github.com/unslothai/unsloth/blob/main/unsloth/models/llama.py unsloth/models/llama.py]
* '''Lines:''' 2578-2800 (FastLlamaModel.get_peft_model)

=== Signature ===
<syntaxhighlight lang="python">
@staticmethod
def get_peft_model(
    model,
    r: int = 16,                    # LoRA rank (8-128 typical)
    target_modules: List[str] = [   # Layers to apply LoRA
        "q_proj", "k_proj", "v_proj", "o_proj",   # Attention
        "gate_proj", "up_proj", "down_proj",      # MLP
    ],
    lora_alpha: int = 16,           # LoRA scaling factor (typically equals r)
    lora_dropout: float = 0.0,      # 0 enables kernel fusions
    bias: str = "none",             # "none" is optimized
    layers_to_transform: List[int] = None,  # Specific layers (None = all)
    layers_pattern: str = None,
    use_gradient_checkpointing: str = "unsloth",  # 30% VRAM reduction
    random_state: int = 3407,
    max_seq_length: int = 2048,     # Not used (deprecated)
    use_rslora: bool = False,       # Rank-stabilized LoRA
    modules_to_save: List[str] = None,  # Full-precision modules
    init_lora_weights: bool = True,
    loftq_config: dict = {},
    temporary_location: str = "_unsloth_temporary_saved_buffers",
    qat_scheme: str = None,         # Quantization-aware training
    **kwargs,
) -> PeftModel:
    """
    Apply LoRA adapters to the model with Unsloth optimizations.

    Returns:
        PeftModel with LoRA adapters attached and optimizations applied.
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel
# get_peft_model is a method of FastLanguageModel
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model || PreTrainedModel || Yes || Model loaded via FastLanguageModel.from_pretrained()
|-
| r || int || No || LoRA rank; higher = more capacity but more memory (default: 16)
|-
| target_modules || List[str] || No || Layer names to apply LoRA (default: attention + MLP)
|-
| lora_alpha || int || No || Scaling factor; typically equals r (default: 16)
|-
| lora_dropout || float || No || Dropout rate; 0.0 enables fused kernels (default: 0.0)
|-
| bias || str || No || Bias training mode: "none", "all", or "lora_only" (default: "none")
|-
| use_gradient_checkpointing || str || No || "unsloth" for optimized checkpointing
|-
| use_rslora || bool || No || Enable Rank-Stabilized LoRA (default: False)
|-
| modules_to_save || List[str] || No || Modules to train in full precision (e.g., ["embed_tokens", "lm_head"])
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| model || PeftModelForCausalLM || Model with LoRA adapters and Unsloth optimizations
|}

== Usage Examples ==

=== Basic LoRA Setup ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# Load model first
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Apply LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                          # LoRA rank
    target_modules=[               # Apply to these layers
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,                 # Scaling factor
    lora_dropout=0,                # 0 enables kernel fusions
    bias="none",                   # "none" is optimized
    use_gradient_checkpointing="unsloth",  # 30% VRAM savings
    random_state=3407,
)

# Model is now ready for training
print(f"Trainable parameters: {model.print_trainable_parameters()}")
</syntaxhighlight>

=== High-Rank LoRA for RL ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-8B-bnb-4bit",
    max_seq_length=4096,
    load_in_4bit=True,
    fast_inference=True,
    max_lora_rank=64,  # Important: match with r below
)

# Higher rank for RL training capacity
model = FastLanguageModel.get_peft_model(
    model,
    r=64,              # Higher rank for RL
    lora_alpha=64,     # Match alpha to rank
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)
</syntaxhighlight>

=== Training Embeddings ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Train embeddings alongside LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "embed_tokens", "lm_head",  # Also train embeddings
    ],
    modules_to_save=["embed_tokens", "lm_head"],  # Full precision
    lora_dropout=0,
    bias="none",
)
# Note: Original embeddings are offloaded to CPU to save VRAM
</syntaxhighlight>

=== Using RSLoRA ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)

# RSLoRA for improved training stability with higher ranks
model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    lora_alpha=32,
    use_rslora=True,  # Rank-stabilized scaling
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:unslothai_unsloth_LoRA_Injection]]

=== Requires Environment ===
* [[requires_env::Environment:unslothai_unsloth_CUDA]]

=== Uses Heuristics ===
* [[uses_heuristic::Heuristic:unslothai_unsloth_LoRA_Rank_Selection]]
