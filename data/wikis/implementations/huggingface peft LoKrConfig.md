{{Implementation
|domain=NLP,Computer Vision,PEFT,Parameter-Efficient Fine-Tuning,LoKr,Low-Rank Adaptation,Kronecker Product
|link=https://github.com/huggingface/peft
}}

== Overview ==

=== Description ===

The <code>LoKrConfig</code> class is a configuration dataclass for Low-Rank Kronecker Product (LoKr) adaptation in PEFT. LoKr is an advanced parameter-efficient fine-tuning method that uses Kronecker product decomposition to create extremely efficient low-rank adaptations of neural network layers.

LoKrConfig inherits from <code>LycorisConfig</code> and provides comprehensive configuration options for LoKr adapters. The method combines ideas from multiple papers including https://huggingface.co/papers/2108.06098 and https://huggingface.co/papers/2309.14859, offering sophisticated control over the decomposition strategy.

Key features include:
* Kronecker product-based low-rank decomposition
* Optional decomposition of both matrices in the Kronecker product
* Configurable decomposition factor
* Rank dropout with optional scaling
* Effective Conv2d decomposition for convolutional layers
* Layer-specific rank and alpha patterns
* Multiple weight initialization strategies including "lycoris" style

=== Usage ===

LoKrConfig is used to configure LoKr adapters before applying them to models via the PEFT library. It's particularly effective for models where extreme parameter efficiency is required while maintaining performance. The Kronecker product decomposition often achieves better compression ratios than standard low-rank methods.

== Code Reference ==

=== Source Location ===
* '''Repository:''' huggingface/peft
* '''File Path:''' <code>src/peft/tuners/lokr/config.py</code>
* '''Lines:''' 23-156

=== Signature ===
<syntaxhighlight lang="python">
@dataclass
class LoKrConfig(LycorisConfig):
    r: int = 8
    alpha: int = 8
    rank_dropout: float = 0.0
    module_dropout: float = 0.0
    use_effective_conv2d: bool = False
    decompose_both: bool = False
    decompose_factor: int = -1
    rank_dropout_scale: bool = False
    target_modules: Optional[Union[list[str], str]] = None
    exclude_modules: Optional[Union[list[str], str]] = None
    init_weights: Union[bool, Literal["lycoris"]] = True
    layers_to_transform: Optional[Union[list[int], int]] = None
    layers_pattern: Optional[Union[list[str], str]] = None
    modules_to_save: Optional[list[str]] = None
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft.tuners.lokr.config import LoKrConfig
# or
from peft import LoKrConfig
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===

{| class="wikitable"
! Parameter !! Type !! Default !! Description
|-
| <code>r</code> || <code>int</code> || <code>8</code> || LoKr rank - controls the dimension of the low-rank decomposition
|-
| <code>alpha</code> || <code>int</code> || <code>8</code> || The alpha parameter for LoKr scaling
|-
| <code>rank_dropout</code> || <code>float</code> || <code>0.0</code> || The dropout probability for rank dimension during training
|-
| <code>module_dropout</code> || <code>float</code> || <code>0.0</code> || The dropout probability for disabling LoKr modules during training
|-
| <code>use_effective_conv2d</code> || <code>bool</code> || <code>False</code> || Use parameter effective decomposition for Conv2d (and Conv1d) with ksize > 1 ("Proposition 3" from FedPara paper)
|-
| <code>decompose_both</code> || <code>bool</code> || <code>False</code> || Perform rank decomposition of left Kronecker product matrix (more parameter efficient)
|-
| <code>decompose_factor</code> || <code>int</code> || <code>-1</code> || Kronecker product decomposition factor. -1 means automatic selection
|-
| <code>rank_dropout_scale</code> || <code>bool</code> || <code>False</code> || Whether to scale the rank dropout while training
|-
| <code>target_modules</code> || <code>Optional[Union[list[str], str]]</code> || <code>None</code> || List of module names or regex expression to replace with LoKr. Can be 'all-linear' for all linear layers except output
|-
| <code>exclude_modules</code> || <code>Optional[Union[list[str], str]]</code> || <code>None</code> || List of module names or regex expression to exclude from LoKr adaptation
|-
| <code>init_weights</code> || <code>Union[bool, Literal["lycoris"]]</code> || <code>True</code> || Whether to initialize weights. Can be True, False, or "lycoris" for LyCORIS-style initialization
|-
| <code>layers_to_transform</code> || <code>Optional[Union[list[int], int]]</code> || <code>None</code> || The layer indices to transform. If specified, only these layers will be adapted
|-
| <code>layers_pattern</code> || <code>Optional[Union[list[str], str]]</code> || <code>None</code> || The layer pattern name (e.g., 'layers' or 'h'), used with layers_to_transform
|-
| <code>modules_to_save</code> || <code>Optional[list[str]]</code> || <code>None</code> || List of modules apart from LoKr layers to be set as trainable and saved
|}

'''Inherited from LycorisConfig:'''
{| class="wikitable"
! Parameter !! Type !! Description
|-
| <code>rank_pattern</code> || <code>dict</code> || Mapping from layer names/regex to ranks different from default r
|-
| <code>alpha_pattern</code> || <code>dict</code> || Mapping from layer names/regex to alphas different from default alpha
|}

=== Outputs ===

{| class="wikitable"
! Attribute !! Type !! Description
|-
| <code>peft_type</code> || <code>PeftType</code> || Automatically set to <code>PeftType.LOKR</code> in <code>__post_init__</code>
|-
| <code>target_modules</code> || <code>Optional[Union[set[str], str]]</code> || Converted to set if originally a list
|-
| <code>exclude_modules</code> || <code>Optional[Union[set[str], str]]</code> || Converted to set if originally a list
|}

== Usage Examples ==

=== Example 1: Basic LoKr Configuration ===
<syntaxhighlight lang="python">
from peft import LoKrConfig, get_peft_model
from transformers import AutoModelForCausalLM

# Create LoKr configuration
config = LoKrConfig(
    r=8,
    alpha=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    rank_dropout=0.0,
    module_dropout=0.0,
    init_weights=True,
)

# Apply to model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model = get_peft_model(model, config)
model.print_trainable_parameters()
</syntaxhighlight>

=== Example 2: LoKr with Both Matrices Decomposed ===
<syntaxhighlight lang="python">
from peft import LoKrConfig, get_peft_model

# More aggressive compression with decompose_both
config = LoKrConfig(
    r=8,
    alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    decompose_both=True,  # Decompose both Kronecker matrices
    decompose_factor=4,  # Specific decomposition factor
    rank_dropout=0.0,
    init_weights=True,
)

model = get_peft_model(base_model, config)

# This configuration uses even fewer parameters than standard LoKr
print("Highly compressed LoKr model ready")
</syntaxhighlight>

=== Example 3: LoKr for Stable Diffusion with Effective Conv2d ===
<syntaxhighlight lang="python">
from diffusers import StableDiffusionPipeline
from peft import LoKrModel, LoKrConfig

# Configuration for text encoder
config_te = LoKrConfig(
    r=8,
    alpha=32,
    target_modules=["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"],
    rank_dropout=0.0,
    module_dropout=0.0,
    init_weights=True,
)

# Configuration for UNet with effective conv2d
config_unet = LoKrConfig(
    r=8,
    alpha=32,
    target_modules=[
        "proj_in",
        "proj_out",
        "to_k",
        "to_q",
        "to_v",
        "to_out.0",
        "ff.net.0.proj",
        "ff.net.2",
    ],
    rank_dropout=0.0,
    module_dropout=0.0,
    init_weights=True,
    use_effective_conv2d=True,  # Important for Conv2d layers
    decompose_both=True,  # Maximum compression
)

model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
model.text_encoder = LoKrModel(model.text_encoder, config_te, "default")
model.unet = LoKrModel(model.unet, config_unet, "default")
</syntaxhighlight>

=== Example 4: LoKr with LyCORIS-Style Initialization ===
<syntaxhighlight lang="python">
from peft import LoKrConfig, get_peft_model

# Use LyCORIS initialization strategy
config = LoKrConfig(
    r=16,
    alpha=32,
    target_modules=[".*attention.*(q|k|v)_proj"],  # Regex pattern
    init_weights="lycoris",  # Use LyCORIS-style initialization
    decompose_both=False,
    rank_dropout=0.1,
    rank_dropout_scale=True,  # Scale dropout during training
)

model = get_peft_model(base_model, config)
</syntaxhighlight>

=== Example 5: LoKr with Layer-Specific Configuration ===
<syntaxhighlight lang="python">
from peft import LoKrConfig, get_peft_model

# Configure with rank patterns and selective layers
config = LoKrConfig(
    r=8,
    alpha=8,
    target_modules=["q_proj", "v_proj"],
    # Only adapt specific layers
    layers_to_transform=[0, 1, 2, 3, 28, 29, 30, 31],  # First 4 and last 4 layers
    layers_pattern="layers",
    # Layer-specific rank overrides
    rank_pattern={
        "^model.layers.[0-3].*": 16,  # Higher rank for first layers
        "^model.layers.(28|29|30|31).*": 16,  # Higher rank for last layers
    },
    alpha_pattern={
        "^model.layers.[0-3].*": 32,
        "^model.layers.(28|29|30|31).*": 32,
    },
    decompose_both=True,
    decompose_factor=-1,  # Auto-select factor
    modules_to_save=["lm_head"],
)

model = get_peft_model(base_model, config)
</syntaxhighlight>

=== Example 6: LoKr with Dropout and Regularization ===
<syntaxhighlight lang="python">
from peft import LoKrConfig, get_peft_model

config = LoKrConfig(
    r=8,
    alpha=16,
    target_modules="all-linear",  # Target all linear layers
    exclude_modules=["lm_head"],  # Exclude output layer
    rank_dropout=0.1,  # 10% rank dropout
    module_dropout=0.05,  # 5% module dropout
    rank_dropout_scale=True,  # Scale activations during dropout
    init_weights=True,
    decompose_both=False,
)

model = get_peft_model(base_model, config)

# Train with regularization
model.train()
# ... training loop ...
</syntaxhighlight>

== Implementation Details ==

=== Kronecker Product Decomposition ===
LoKr uses Kronecker product factorization: <code>W ≈ A ⊗ B</code> where:
* <code>W</code> is the original weight matrix
* <code>A</code> and <code>B</code> are smaller matrices
* <code>⊗</code> denotes the Kronecker product

When <code>decompose_both=True</code>, both A and B are further decomposed into low-rank matrices, achieving even higher compression.

=== Decomposition Factor ===
The <code>decompose_factor</code> parameter controls how the matrices are split:
* <code>-1</code> (default): Automatic selection based on dimensions
* Positive integer: Specific factor for decomposition
* Affects the balance between the two Kronecker factors

=== Rank Dropout Scaling ===
When <code>rank_dropout_scale=True</code>, the implementation scales the remaining activations to maintain expected values during dropout, similar to standard dropout behavior. This can improve training stability.

=== Post-Initialization Processing ===
The <code>__post_init__</code> method:
* Sets <code>peft_type</code> to <code>PeftType.LOKR</code>
* Converts <code>target_modules</code> and <code>exclude_modules</code> to sets
* Validates <code>layers_pattern</code> is only used with <code>layers_to_transform</code>

=== Initialization Modes ===
* <code>True</code>: Standard PyTorch initialization
* <code>False</code>: No initialization (not recommended)
* <code>"lycoris"</code>: Specialized initialization from LyCORIS repository

== Related Pages ==

* [[huggingface_peft_LoKrModel|LoKrModel]] - Model class that applies LoKr adapters
* [[huggingface_peft_LoKrLayer|LoKrLayer]] - Layer implementation for LoKr
* [[huggingface_peft_LycorisConfig|LycorisConfig]] - Base configuration class for Lycoris methods
* [[huggingface_peft_LoHaConfig|LoHaConfig]] - Configuration for LoHa (related method)
* [[huggingface_peft_LoRAConfig|LoRAConfig]] - Configuration for standard LoRA
* [[huggingface_peft_get_peft_model|get_peft_model]] - Function to create PEFT models

[[Category:PEFT]]
[[Category:Configuration]]
[[Category:LoKr]]
[[Category:Kronecker Product]]
[[Category:Low-Rank Adaptation]]
[[Category:Parameter-Efficient Fine-Tuning]]
