{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Paper|AdaLoRA|https://arxiv.org/abs/2303.10512]]
|-
! Domains
| [[domain::NLP]], [[domain::PEFT]], [[domain::Adaptive_Rank]]
|-
! Last Updated
| [[last_updated::2025-12-18 14:00 GMT]]
|}

== Overview ==

Configuration class for AdaLoRA that stores parameters for adaptive rank allocation during LoRA fine-tuning, including training schedule and sensitivity-based rank budgeting.

=== Description ===

AdaLoraConfig extends LoraConfig to support three-phase training: initial warmup (tinit steps), rank reduction phase, and final fine-tuning (tfinal steps). The init_r parameter sets starting rank per layer, while target_r sets the average target rank after reduction. RankAllocator uses EMA-smoothed sensitivity scores (beta1, beta2) with orthogonal regularization (orth_reg_weight) to allocate ranks across layers.

=== Usage ===

Use AdaLoraConfig when you want automatic rank allocation based on layer importance. Must specify total_step for the training schedule. Note that the standard `r` parameter is ignored - use init_r instead. DoRA and LoftQ are not supported with AdaLoRA.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft PEFT]
* '''File:''' [https://github.com/huggingface/peft/blob/main/src/peft/tuners/adalora/config.py src/peft/tuners/adalora/config.py]
* '''Lines:''' 1-109

=== Signature ===
<syntaxhighlight lang="python">
@dataclass
class AdaLoraConfig(LoraConfig):
    """
    Configuration for AdaLoRA adaptive rank allocation.

    Args:
        target_r: Target average rank after reduction
        init_r: Initial rank for each matrix
        tinit: Initial warmup steps (no rank reduction)
        tfinal: Final fine-tuning steps (no rank reduction)
        deltaT: Steps between budget allocations
        beta1: EMA hyperparameter for sensitivity smoothing
        beta2: EMA hyperparameter for uncertainty
        orth_reg_weight: Orthogonal regularization coefficient
        total_step: Total training steps (required)
        rank_pattern: Saved rank allocation pattern
    """
    target_r: int = 8
    init_r: int = 12
    tinit: int = 0
    tfinal: int = 0
    deltaT: int = 1
    beta1: float = 0.85
    beta2: float = 0.85
    orth_reg_weight: float = 0.5
    total_step: Optional[int] = None
    rank_pattern: Optional[dict] = None
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft import AdaLoraConfig
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| target_r || int || No || Target average rank (default: 8)
|-
| init_r || int || No || Initial rank per layer (default: 12)
|-
| total_step || int || Yes || Total training steps (required)
|-
| tinit || int || No || Warmup steps before rank reduction
|-
| tfinal || int || No || Final fine-tuning steps
|-
| target_modules || list[str] || No || Modules to apply AdaLoRA
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| AdaLoraConfig || dataclass || Configuration object for get_peft_model
|}

== Usage Examples ==

=== Basic AdaLoRA Configuration ===
<syntaxhighlight lang="python">
from peft import AdaLoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

config = AdaLoraConfig(
    init_r=12,              # Start with rank 12
    target_r=4,             # Reduce to average rank 4
    total_step=10000,       # Required: total training steps
    target_modules=["q_proj", "k_proj", "v_proj"],
)

model = get_peft_model(model, config)
</syntaxhighlight>

=== AdaLoRA with Training Schedule ===
<syntaxhighlight lang="python">
from peft import AdaLoraConfig

# Three-phase training:
# 1. Steps 0-100: Warmup (no rank reduction)
# 2. Steps 100-900: Rank reduction phase
# 3. Steps 900-1000: Final fine-tuning (no reduction)

config = AdaLoraConfig(
    init_r=16,
    target_r=8,
    tinit=100,              # 100 steps warmup
    tfinal=100,             # 100 steps final tuning
    total_step=1000,        # Total training steps
    deltaT=10,              # Allocate ranks every 10 steps
    beta1=0.85,             # Sensitivity EMA
    beta2=0.85,             # Uncertainty EMA
    orth_reg_weight=0.5,    # Orthogonal regularization
    target_modules=["q_proj", "v_proj"],
)
</syntaxhighlight>

=== Validation Rules ===
<syntaxhighlight lang="python">
# AdaLoRA requires total_step
config = AdaLoraConfig(target_r=8)  # Raises ValueError

# Schedule must allow budgeting phase
config = AdaLoraConfig(
    tinit=500,
    tfinal=500,
    total_step=1000,  # Raises ValueError: no budgeting phase
)

# DoRA not supported
config = AdaLoraConfig(
    total_step=1000,
    use_dora=True,  # Raises ValueError
)
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:huggingface_peft_Core_Environment]]
