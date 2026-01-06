{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
|-
! Domains
| [[domain::NLP]], [[domain::PEFT]], [[domain::Fourier_Adaptation]]
|-
! Last Updated
| [[last_updated::2025-12-18 14:00 GMT]]
|}

== Overview ==

Configuration class for FourierFT that stores parameters for Discrete Fourier Transform-based parameter-efficient fine-tuning in the frequency domain.

=== Description ===

FourierFTConfig stores configuration for Fourier-based adaptation. The n_frequency parameter controls how many spectral components are learned (must be ≤ d^2 for d×d weights). The scaling parameter is analogous to LoRA's lora_alpha. For similar quality to LoRA r=8, use n_frequency=1000 (about 16x fewer parameters).

=== Usage ===

Use FourierFTConfig for frequency-domain adaptation with extreme parameter efficiency. FourierFT requires about 10-16x fewer parameters than LoRA for similar performance. Use higher n_frequency for better accuracy at cost of memory.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft PEFT]
* '''File:''' [https://github.com/huggingface/peft/blob/main/src/peft/tuners/fourierft/config.py src/peft/tuners/fourierft/config.py]
* '''Lines:''' 1-207

=== Signature ===
<syntaxhighlight lang="python">
@dataclass
class FourierFTConfig(PeftConfig):
    """
    Configuration for FourierFT (Fourier Fine-Tuning).

    Args:
        n_frequency: Number of learnable frequencies (1-d^2)
        scaling: Scaling factor (like lora_alpha)
        random_loc_seed: Seed for spectral entry locations
        target_modules: Modules to apply FourierFT to
        fan_in_fan_out: True for Conv1D layers
        bias: Bias handling ('none', 'all', 'fourier_only')
        init_weights: True for zeros, False for normal distribution
        n_frequency_pattern: Per-layer n_frequency overrides
    """
    n_frequency: int = 1000
    scaling: float = 150.0
    random_loc_seed: Optional[int] = 777
    target_modules: Optional[Union[list[str], str]] = None
    fan_in_fan_out: bool = False
    bias: str = "none"
    init_weights: bool = False
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft import FourierFTConfig
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| n_frequency || int || No || Number of frequencies (default: 1000)
|-
| scaling || float || No || Scaling factor (default: 150.0)
|-
| target_modules || list[str] || No || Modules to adapt
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| FourierFTConfig || dataclass || Configuration for get_peft_model
|}

== Usage Examples ==

=== Basic FourierFT Configuration ===
<syntaxhighlight lang="python">
from peft import FourierFTConfig, get_peft_model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

config = FourierFTConfig(
    n_frequency=1000,
    scaling=300.0,           # Recommended for LLaMA
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(model, config)
</syntaxhighlight>

=== Recommended Settings by Task ===
<syntaxhighlight lang="python">
from peft import FourierFTConfig

# NLU tasks (RoBERTa)
config_nlu = FourierFTConfig(
    n_frequency=1000,
    scaling=150.0,
    target_modules=["query", "value"],
)

# Instruction tuning (LLaMA)
config_llm = FourierFTConfig(
    n_frequency=1000,
    scaling=300.0,
    target_modules=["q_proj", "v_proj"],
)

# Image classification (ViT)
config_vit = FourierFTConfig(
    n_frequency=3000,
    scaling=300.0,
    target_modules=["query", "value"],
)
</syntaxhighlight>

=== Parameter Comparison ===
<syntaxhighlight lang="python">
# LoRA r=8 on layer d=4096:
# Params = 2 * d * r = 2 * 4096 * 8 = 65,536

# FourierFT n_frequency=1000:
# Params = 1000 (per layer)
# About 65x fewer parameters!
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:huggingface_peft_Core_Environment]]
