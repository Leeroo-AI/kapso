= PolyConfig =

== Knowledge Sources ==
* Source Repository: https://github.com/huggingface/peft
* Paper: [https://huggingface.co/papers/2202.13914 Polytropon (Poly)]
* Paper: [https://huggingface.co/papers/2211.03831 Multi-Head Routing (MHR)]

== Domains ==
* [[NLP]]
* [[PEFT]] (Parameter-Efficient Fine-Tuning)
* [[Multi-Task Learning]]
* [[LoRA]] (Low-Rank Adaptation)
* [[Model Adaptation]]

== Overview ==

=== Description ===
PolyConfig is the configuration class for storing the configuration of a PolyModel. It implements the Polytropon (Poly) and Multi-Head Routing (MHR) parameter-efficient fine-tuning approaches, which use multiple LoRA (Low-Rank Adaptation) modules ("skills") that can be combined dynamically for multi-task scenarios.

The Poly approach enables efficient multi-task learning by maintaining a pool of LoRA modules that can be routed and combined based on task requirements. Multi-Head Routing (MHR) extends this by allowing splits within each LoRA module for more fine-grained control.

=== Usage ===
PolyConfig is used to configure Poly layers in a model for multi-task learning scenarios. It specifies the number of tasks, skills (LoRA modules), and splits for routing, along with standard LoRA parameters like rank and target modules.

== Code Reference ==

=== Source Location ===
<code>src/peft/tuners/poly/config.py</code>

=== Signature ===
<syntaxhighlight lang="python">
@dataclass
class PolyConfig(PeftConfig):
    def __init__(
        self,
        r: int = 8,
        target_modules: Optional[Union[list[str], str]] = None,
        exclude_modules: Optional[Union[list[str], str]] = None,
        modules_to_save: Optional[list[str]] = None,
        init_weights: bool = True,
        poly_type: Literal["poly"] = "poly",
        n_tasks: int = 1,
        n_skills: int = 4,
        n_splits: int = 1,
    )
</syntaxhighlight>

=== Import Statement ===
<syntaxhighlight lang="python">
from peft.tuners.poly.config import PolyConfig
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
! Parameter !! Type !! Default !! Description
|-
| r || int || 8 || Attention dimension of each LoRA in Poly
|-
| target_modules || Optional[Union[list[str], str]] || None || The names of the modules to apply Poly to (e.g., ['q', 'v'] or regex pattern)
|-
| exclude_modules || Optional[Union[list[str], str]] || None || The names of the modules to exclude from Poly (regex match or exact match)
|-
| modules_to_save || Optional[list[str]] || None || List of modules apart from Poly layers to be set as trainable and saved
|-
| init_weights || bool || True || Whether to initialize the weights of the Poly layers
|-
| poly_type || Literal["poly"] || "poly" || The variant of the Poly module to use (currently only "poly" is supported)
|-
| n_tasks || int || 1 || The number of tasks in a multitasking scenario
|-
| n_skills || int || 4 || The number of skills (LoRA) in each Poly layer
|-
| n_splits || int || 1 || The number of splits within each LoRA (values > 1 indicate Multi-Head Routing)
|}

=== Outputs ===
{| class="wikitable"
! Return Type !! Description
|-
| PolyConfig || A configured PolyConfig instance with peft_type set to PeftType.POLY
|}

=== Post-Initialization ===
The <code>__post_init__</code> method:
* Sets <code>peft_type</code> to <code>PeftType.POLY</code>
* Converts <code>target_modules</code> to a set if provided as a list
* Converts <code>exclude_modules</code> to a set if provided as a list

== Usage Examples ==

=== Basic Multi-Task Configuration ===
<syntaxhighlight lang="python">
from peft import PolyConfig, get_peft_model
from transformers import AutoModelForCausalLM

# Load base model
model = AutoModelForCausalLM.from_pretrained("base-model-name")

# Configure Poly for 3 tasks with 4 skills per layer
config = PolyConfig(
    r=8,
    target_modules=["q_proj", "v_proj"],
    n_tasks=3,
    n_skills=4,
    n_splits=1
)

# Apply Poly to model
peft_model = get_peft_model(model, config)
</syntaxhighlight>

=== Multi-Head Routing Configuration ===
<syntaxhighlight lang="python">
from peft import PolyConfig, get_peft_model

# Configure Poly with Multi-Head Routing (MHR)
config = PolyConfig(
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    n_tasks=5,
    n_skills=8,
    n_splits=4,  # Use 4 splits for MHR
    init_weights=True
)

peft_model = get_peft_model(model, config)
</syntaxhighlight>

=== Regex Pattern for Target Modules ===
<syntaxhighlight lang="python">
from peft import PolyConfig

# Use regex to target specific modules
config = PolyConfig(
    r=8,
    target_modules=".*decoder.*(SelfAttention|EncDecAttention).*(q|v)$",
    exclude_modules=["classifier"],
    n_tasks=10,
    n_skills=6,
    modules_to_save=["classifier"]
)
</syntaxhighlight>

== Related Pages ==
* [[huggingface_peft_PolyModel|PolyModel]] - The model implementation that uses PolyConfig
* [[huggingface_peft_PolyRouter|PolyRouter]] - The routing mechanism for Poly layers
* [[huggingface_peft_PolyLayer|PolyLayer]] - The layer implementation for Poly
* [[huggingface_peft_LoraConfig|LoraConfig]] - Related LoRA configuration
* [[PEFT]] - Parameter-Efficient Fine-Tuning overview
* [[Multi-Task Learning]] - Multi-task learning concepts

== Categories ==
[[Category:PEFT]]
[[Category:Configuration]]
[[Category:Multi-Task Learning]]
[[Category:LoRA]]
[[Category:HuggingFace]]
