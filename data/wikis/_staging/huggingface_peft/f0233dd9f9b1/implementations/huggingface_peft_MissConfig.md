= MiSS Configuration =

== Knowledge Sources ==
* '''Repository:''' [https://github.com/huggingface/peft HuggingFace PEFT]
* '''Source File:''' src/peft/tuners/miss/config.py
* '''Paper:''' [https://huggingface.co/papers/2409.15371 MiSS: Householder Reflection Adaptation]

== Domains ==
* [[Natural Language Processing]]
* [[Parameter-Efficient Fine-Tuning]]
* [[Model Configuration]]
* [[Householder Reflection]]

== Overview ==

=== Description ===
The MiSS (Memory-efficient Incremental Singular Space) configuration class defines the settings for applying MiSS adapters to neural network models. MiSS is a parameter-efficient fine-tuning method based on Householder reflections, providing an alternative to LoRA with different trade-offs between parameter efficiency and model expressiveness.

Key features of MiSS:
* '''Dual-rank decomposition''': Separate ranks for in_features (r) and out_features (mini_r) dimensions
* '''Multiple initialization modes''': balance (default/efficient), bat (nonlinear updates), mini (smaller rank)
* '''Pattern-based targeting''': Flexible module selection with regex support
* '''Layer-specific application''': Optional layer indices for selective adaptation
* '''Dropout support''': Regularization through miss_dropout

The configuration supports three initialization variants:
1. '''balance''' (default=True): Most efficient and general method
2. '''bat''': Enables nonlinear updates across different shards
3. '''mini''': Smaller rank variant for fewer trainable parameters

=== Usage ===
MissConfig is used to configure MiSS adapters when fine-tuning models with PEFT. It inherits from PeftConfig and provides MiSS-specific parameters for controlling the adapter behavior.

== Code Reference ==

=== Source Location ===
<code>src/peft/tuners/miss/config.py</code>

=== Class Signature ===
<syntaxhighlight lang="python">
@dataclass
class MissConfig(PeftConfig):
    r: int = field(default=64)
    miss_dropout: float = field(default=0.0)
    mini_r: int = field(default=1)
    target_modules: Optional[Union[list[str], str]] = field(default=None)
    exclude_modules: Optional[Union[list[str], str]] = field(default=None)
    init_weights: bool | Literal["bat", "mini"] = field(default=True)
    layers_to_transform: Optional[Union[list[int], int]] = field(default=None)
    layers_pattern: Optional[str] = field(default=None)
    bias: str = field(default="none")
    modules_to_save: Optional[list[str]] = field(default=None)
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft.tuners.miss.config import MissConfig
</syntaxhighlight>

== I/O Contract ==

=== Configuration Fields ===
{| class="wikitable"
! Field !! Type !! Default !! Description
|-
| r || int || 64 || Rank for low-rank decomposition along in_features dimension (should be even)
|-
| miss_dropout || float || 0.0 || Dropout probability for MiSS layers
|-
| mini_r || int || 1 || Rank for decomposition along out_features dimension (out_features should be divisible by mini_r)
|-
| target_modules || Optional[Union[list[str], str]] || None || Module names or regex to apply adapter; "all-linear" for all linear layers
|-
| exclude_modules || Optional[Union[list[str], str]] || None || Module names or regex to exclude from adaptation
|-
| init_weights || bool &#124; Literal["bat", "mini"] || True || Initialization mode: True=balance, "bat"=nonlinear updates, "mini"=smaller rank
|-
| layers_to_transform || Optional[Union[list[int], int]] || None || Layer indices to transform; None transforms all
|-
| layers_pattern || Optional[str] || None || Layer pattern name (used with layers_to_transform)
|-
| bias || str || "none" || Bias type: "none", "all", or "MiSS_only"
|-
| modules_to_save || Optional[list[str]] || None || Additional modules to train and save (e.g., classifier heads)
|}

=== Validation Rules ===
The <code>__post_init__</code> method enforces:
* Sets <code>peft_type = PeftType.MISS</code>
* Converts list target_modules and exclude_modules to sets
* Raises ValueError if target_modules is str and layers_to_transform is not None
* Raises ValueError if target_modules is str and layers_pattern is not None

== Usage Examples ==

=== Basic MiSS Configuration ===
<syntaxhighlight lang="python">
from peft import MissConfig, get_peft_model
from transformers import AutoModelForCausalLM

# Load base model
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Create MiSS config with default settings
config = MissConfig(
    r=64,  # Rank along in_features
    target_modules=["c_attn", "c_proj"],  # Target attention layers
    init_weights=True,  # Use balanced initialization
    bias="none",
)

# Apply MiSS to model
peft_model = get_peft_model(model, config)
print(f"Trainable parameters: {peft_model.num_parameters(only_trainable=True)}")
</syntaxhighlight>

=== Mini Variant for Maximum Efficiency ===
<syntaxhighlight lang="python">
# Mini variant uses smaller rank for fewer parameters
config = MissConfig(
    r=32,  # Reduced rank
    mini_r=4,  # Smaller out_features rank
    miss_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
    init_weights="mini",  # Mini initialization
)

# Important: Ensure out_features % mini_r == 0
# For a layer with out_features=768, mini_r should divide 768
# Valid: 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, etc.

peft_model = get_peft_model(model, config)
</syntaxhighlight>

=== Bat Variant for Nonlinear Updates ===
<syntaxhighlight lang="python">
# Bat mode enables nonlinear updates across shards
config = MissConfig(
    r=64,
    target_modules=["attn.c_attn"],
    init_weights="bat",  # Bat initialization
    miss_dropout=0.05,
)

peft_model = get_peft_model(model, config)
</syntaxhighlight>

=== Using Regex Patterns for Target Modules ===
<syntaxhighlight lang="python">
# Use regex to target modules
config = MissConfig(
    r=64,
    # Match all query and value projections in any layer
    target_modules=r".*\.(q_proj|v_proj)$",
    # Exclude specific layers
    exclude_modules=["lm_head", "embed_tokens"],
    init_weights=True,
)

peft_model = get_peft_model(model, config)
</syntaxhighlight>

=== Target All Linear Layers ===
<syntaxhighlight lang="python">
# Apply MiSS to all linear layers except output
config = MissConfig(
    r=32,
    mini_r=2,
    target_modules="all-linear",
    init_weights=True,
)

peft_model = get_peft_model(model, config)
</syntaxhighlight>

=== Layer-Specific Configuration ===
<syntaxhighlight lang="python">
# Apply MiSS only to specific layers
config = MissConfig(
    r=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    layers_to_transform=[0, 1, 2],  # Only first 3 layers
    layers_pattern="layers",  # Pattern for layer indexing
    init_weights=True,
)

peft_model = get_peft_model(model, config)
</syntaxhighlight>

=== Configuration for Sequence Classification ===
<syntaxhighlight lang="python">
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

# Configure MiSS with classifier head training
config = MissConfig(
    r=16,
    mini_r=2,
    target_modules=["query", "value"],
    modules_to_save=["classifier"],  # Also train the classifier
    bias="MiSS_only",  # Add bias only to MiSS layers
    init_weights=True,
)

peft_model = get_peft_model(model, config)
</syntaxhighlight>

=== Configuration with Dropout ===
<syntaxhighlight lang="python">
# Add dropout for regularization
config = MissConfig(
    r=64,
    mini_r=4,
    miss_dropout=0.1,  # 10% dropout
    target_modules=["attn.c_attn", "attn.c_proj", "mlp.c_fc"],
    init_weights=True,
)

peft_model = get_peft_model(model, config)
</syntaxhighlight>

=== Complete Configuration Example ===
<syntaxhighlight lang="python">
from peft import MissConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

config = MissConfig(
    # Core MiSS parameters
    r=64,  # In-features rank (use even number)
    mini_r=4,  # Out-features rank (divisor of out_features)
    miss_dropout=0.05,

    # Target configuration
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    exclude_modules=["lm_head"],

    # Initialization
    init_weights=True,  # Balanced/efficient initialization

    # Bias configuration
    bias="none",

    # Task type
    task_type=TaskType.CAUSAL_LM,
)

peft_model = get_peft_model(model, config)

# Print configuration
print(config)
</syntaxhighlight>

== Related Pages ==
* [[MiSS Model]]
* [[PEFT Configuration]]
* [[Parameter-Efficient Fine-Tuning]]
* [[Householder Reflection]]
* [[LoRA Configuration]]
* [[Model Adapter Methods]]
* [[Low-Rank Decomposition]]

== Notes ==

=== Important Constraints ===
* '''Even rank (r)''': It's best to set r to an even number; otherwise, the default initialization may not work
* '''mini_r divisibility''': out_features should be divisible by mini_r for mini initialization
* '''Regex restrictions''': When target_modules is a regex string:
** layers_to_transform must be None
** layers_pattern must be None

=== Initialization Modes ===
* '''True (balance)''': Default efficient method, suitable for most use cases
* '''"bat"''': Enables nonlinear updates across shards, may provide better expressiveness
* '''"mini"''': Smallest parameter count, recommended to keep out_features % mini_r == 0

=== Bias Options ===
* '''"none"''': No bias parameters (most efficient)
* '''"all"''': Add bias to all adapted layers
* '''"MiSS_only"''': Add bias only to MiSS adapter layers

=== Target Module Selection ===
* '''Explicit list''': ["q_proj", "v_proj"] matches exact names
* '''Regex string''': ".*attn.*" matches all modules containing "attn"
* '''"all-linear"''': Matches all nn.Linear layers except output layer
* '''Exclude modules''': Takes precedence over target_modules

== Advanced Configuration ==

=== Comparing Initialization Modes ===
<syntaxhighlight lang="python">
# Balanced (default) - most efficient
config_balance = MissConfig(r=64, mini_r=1, init_weights=True)

# Bat - nonlinear updates
config_bat = MissConfig(r=64, mini_r=1, init_weights="bat")

# Mini - smallest parameters
config_mini = MissConfig(r=32, mini_r=8, init_weights="mini")

# Compare parameter counts
for name, cfg in [("balance", config_balance), ("bat", config_bat), ("mini", config_mini)]:
    model_copy = get_peft_model(base_model, cfg)
    params = model_copy.num_parameters(only_trainable=True)
    print(f"{name}: {params:,} trainable parameters")
</syntaxhighlight>

=== Dynamic Configuration Based on Model ===
<syntaxhighlight lang="python">
def create_miss_config(model, efficiency="balanced"):
    """Create MiSS config based on model architecture"""

    # Get hidden size to determine ranks
    hidden_size = model.config.hidden_size

    if efficiency == "high":
        r = 16
        mini_r = max(1, hidden_size // 128)
        init_weights = "mini"
    elif efficiency == "balanced":
        r = 64
        mini_r = 1
        init_weights = True
    else:  # "expressive"
        r = 128
        mini_r = 1
        init_weights = "bat"

    return MissConfig(
        r=r,
        mini_r=mini_r,
        init_weights=init_weights,
        target_modules="all-linear",
        miss_dropout=0.05,
    )

config = create_miss_config(model, efficiency="balanced")
</syntaxhighlight>

== References ==
* MiSS Paper: https://huggingface.co/papers/2409.15371
* PEFT Documentation: https://huggingface.co/docs/peft
* Householder Reflection: https://en.wikipedia.org/wiki/Householder_transformation
* Parameter-Efficient Fine-Tuning: https://arxiv.org/abs/2110.04366
