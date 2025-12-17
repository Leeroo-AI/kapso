# Implementation: unslothai_unsloth_get_peft_model

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Paper|LoRA: Low-Rank Adaptation|https://arxiv.org/abs/2106.09685]]
* [[source::Doc|PEFT Docs|https://huggingface.co/docs/peft]]
|-
! Domains
| [[domain::NLP]], [[domain::Training]], [[domain::Parameter_Efficient]]
|-
! Last Updated
| [[last_updated::2025-12-17 15:00 GMT]]
|}

== Overview ==

Concrete tool for injecting LoRA (Low-Rank Adaptation) adapters into a quantized model for parameter-efficient fine-tuning.

=== Description ===

`FastLanguageModel.get_peft_model` transforms a base model into a PEFT model by:

1. **Injecting LoRA adapters** into specified target modules (attention projections, MLP layers)
2. **Configuring optimized gradient checkpointing** using Unsloth's memory-efficient implementation
3. **Setting up mixed-precision training** for embedding layers if they're being trained
4. **Registering Triton kernels** for fast LoRA forward operations

The default configuration targets all attention and MLP layers for maximum expressiveness while maintaining efficiency.

=== Usage ===

Use this after loading a model with `FastLanguageModel.from_pretrained`. This is the step that makes the model trainable by adding small, learnable adapter weights.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unsloth]
* '''File:''' unsloth/models/llama.py
* '''Lines:''' L2578-3100

=== Signature ===
<syntaxhighlight lang="python">
class FastLlamaModel:
    @staticmethod
    def get_peft_model(
        model,
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
        max_seq_length: int = 2048,  # Not used anymore
        use_rslora: bool = False,
        modules_to_save: Optional[List[str]] = None,
        init_lora_weights: bool = True,
        loftq_config: dict = {},
        **kwargs,
    ) -> PeftModel:
        """
        Add LoRA adapters to the model for parameter-efficient fine-tuning.

        Args:
            model: Model from FastLanguageModel.from_pretrained
            r: LoRA rank (higher = more capacity but more memory)
            target_modules: Which layers to add LoRA to
            lora_alpha: LoRA scaling factor (typically = r)
            lora_dropout: Dropout for LoRA layers (0.0 recommended for speed)
            bias: Bias training mode ("none", "all", "lora_only")
            use_gradient_checkpointing: "unsloth" for optimized implementation
            use_rslora: Use Rank-Stabilized LoRA scaling
            modules_to_save: Additional modules to train fully (e.g., "embed_tokens", "lm_head")

        Returns:
            PeftModel with LoRA adapters attached
        """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel
# get_peft_model is a method on FastLanguageModel class
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model || PeftModel || Yes || Model from `FastLanguageModel.from_pretrained`
|-
| r || int || No (default: 16) || LoRA rank - controls adapter capacity
|-
| target_modules || List[str] || No || Layers to add LoRA adapters to
|-
| lora_alpha || int || No (default: 16) || Scaling factor (effective lr = lora_alpha/r * base_lr)
|-
| lora_dropout || float || No (default: 0.0) || Dropout probability (0.0 for fastest training)
|-
| use_gradient_checkpointing || str || No (default: "unsloth") || Checkpointing strategy
|-
| use_rslora || bool || No (default: False) || Use rank-stabilized scaling
|-
| modules_to_save || List[str] || No || Modules to train fully (e.g., ["embed_tokens", "lm_head"])
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| model || PeftModel || Model with LoRA adapters injected and ready for training
|}

== Usage Examples ==

=== Standard LoRA Configuration ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# Load model first
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct",
    max_seq_length = 2048,
    load_in_4bit = True,
)

# Add LoRA adapters with default settings
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,              # Rank 16 - good balance
    lora_alpha = 16,     # Alpha = r for stable training
    lora_dropout = 0,    # No dropout for speed
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    use_gradient_checkpointing = "unsloth",  # Optimized checkpointing
    random_state = 3407,
)

# Check trainable parameters
model.print_trainable_parameters()
# Output: trainable params: 3,407,872 || all params: 1,239,221,248 || trainable%: 0.275
</syntaxhighlight>

=== Higher Rank for Complex Tasks ===
<syntaxhighlight lang="python">
# For tasks requiring more capacity (math, code, reasoning)
model = FastLanguageModel.get_peft_model(
    model,
    r = 64,              # Higher rank for more capacity
    lora_alpha = 64,     # Keep alpha = r
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    use_gradient_checkpointing = "unsloth",
)
</syntaxhighlight>

=== Training Embeddings ===
<syntaxhighlight lang="python">
# For tasks requiring vocabulary adaptation (new languages, domains)
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    lora_alpha = 16,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    # Train embedding and output layers fully
    modules_to_save = ["embed_tokens", "lm_head"],
    use_gradient_checkpointing = "unsloth",
)
</syntaxhighlight>

=== Selective Layer Training ===
<syntaxhighlight lang="python">
# Only train specific layers (e.g., last 8 layers)
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    lora_alpha = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    layers_to_transform = list(range(24, 32)),  # Last 8 layers of 32-layer model
    use_gradient_checkpointing = "unsloth",
)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:unslothai_unsloth_LoRA_Configuration]]

=== Requires Environment ===
* [[requires_env::Environment:unslothai_unsloth_CUDA]]

