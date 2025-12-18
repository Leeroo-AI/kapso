{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Paper|BOFT|https://arxiv.org/abs/2311.06243]]
|-
! Domains
| [[domain::NLP]], [[domain::PEFT]], [[domain::Orthogonal_Fine_Tuning]]
|-
! Last Updated
| [[last_updated::2025-12-18 14:00 GMT]]
|}

== Overview ==

Configuration class for BOFT (Butterfly Orthogonal Fine-Tuning) that stores parameters for block-diagonal orthogonal transformations via butterfly factorization.

=== Description ===

BOFTConfig stores configuration for butterfly-factorized orthogonal fine-tuning. Key parameters are boft_block_size (or boft_block_num) which control the orthogonal block dimensions, and boft_n_butterfly_factor which determines the number of butterfly factors (1 = vanilla OFT). Only one of block_size or block_num can be specified since block_size * block_num = layer_dimension.

=== Usage ===

Use BOFTConfig for orthogonal fine-tuning with butterfly factorization. BOFT provides more parameter efficiency than vanilla OFT via butterfly decomposition. Supports Linear and Conv2d layers. Set boft_n_butterfly_factor > 1 to increase effective block size while reducing parameters.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft PEFT]
* '''File:''' [https://github.com/huggingface/peft/blob/main/src/peft/tuners/boft/config.py src/peft/tuners/boft/config.py]
* '''Lines:''' 1-161

=== Signature ===
<syntaxhighlight lang="python">
@dataclass
class BOFTConfig(PeftConfig):
    """
    Configuration for BOFT (Butterfly Orthogonal Fine-Tuning).

    Args:
        boft_block_size: Block size (mutually exclusive with block_num)
        boft_block_num: Number of blocks (mutually exclusive with block_size)
        boft_n_butterfly_factor: Number of butterfly factors (1 = vanilla OFT)
        target_modules: Modules to apply BOFT to
        boft_dropout: Multiplicative dropout probability
        fan_in_fan_out: True for Conv1D layers (e.g., GPT-2)
        bias: Bias handling ('none', 'all', 'boft_only')
        modules_to_save: Additional modules to train/save
    """
    boft_block_size: int = 4
    boft_block_num: int = 0
    boft_n_butterfly_factor: int = 1
    target_modules: Optional[Union[list[str], str]] = None
    boft_dropout: float = 0.0
    fan_in_fan_out: bool = False
    bias: str = "none"
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft import BOFTConfig
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| boft_block_size || int || No || Block size (default: 4, exclusive with block_num)
|-
| boft_block_num || int || No || Number of blocks (default: 0, exclusive with block_size)
|-
| boft_n_butterfly_factor || int || No || Butterfly factors (default: 1)
|-
| target_modules || list[str] || No || Modules to adapt
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| BOFTConfig || dataclass || Configuration for get_peft_model
|}

== Usage Examples ==

=== Basic BOFT Configuration ===
<syntaxhighlight lang="python">
from peft import BOFTConfig, get_peft_model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

config = BOFTConfig(
    boft_block_size=8,
    boft_n_butterfly_factor=1,  # Vanilla OFT
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(model, config)
</syntaxhighlight>

=== BOFT with Butterfly Factorization ===
<syntaxhighlight lang="python">
from peft import BOFTConfig

# More butterfly factors = larger effective block, fewer params
config = BOFTConfig(
    boft_block_size=4,
    boft_n_butterfly_factor=2,  # Effective block size doubles
    boft_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj"],
)
</syntaxhighlight>

=== Using Block Num Instead ===
<syntaxhighlight lang="python">
from peft import BOFTConfig

# Specify number of blocks instead of size
# block_size * block_num = layer_dimension
config = BOFTConfig(
    boft_block_num=128,  # Number of blocks
    target_modules=["q_proj"],
)
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:huggingface_peft_Core_Environment]]
