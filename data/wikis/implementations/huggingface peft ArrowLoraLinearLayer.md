{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Paper|Arrow|https://arxiv.org/abs/2404.15198]]
|-
! Domains
| [[domain::NLP]], [[domain::PEFT]], [[domain::Mixture_of_Experts]]
|-
! Last Updated
| [[last_updated::2025-12-18 14:00 GMT]]
|}

== Overview ==

Arrow routing layer that performs mixture-of-experts style routing over multiple LoRA adapters using prototype-based cosine similarity for expert selection.

=== Description ===

ArrowLoraLinearLayer implements the Arrow algorithm which routes tokens to the most relevant LoRA adapters based on cosine similarity with learned prototypes. For each adapter, a prototype vector is computed via SVD of the LoRA weight matrices. During inference, tokens are matched to prototypes using top-k selection, and expert outputs are combined via weighted sum. The layer also supports General Knowledge Subtraction (GKS) to purify task-specific adapters by removing shared knowledge.

=== Usage ===

Use Arrow when you have multiple task-specific LoRA adapters and want token-level routing to appropriate experts. Arrow is particularly effective for multi-task scenarios where different parts of the input may benefit from different specializations. The `create_arrow_model` helper loads adapters and sets up routing automatically.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft PEFT]
* '''File:''' [https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/arrow.py src/peft/tuners/lora/arrow.py]
* '''Lines:''' 1-477

=== Signature ===
<syntaxhighlight lang="python">
class ArrowLoraLinearLayer(nn.Module):
    """
    Arrow routing algorithm for LoRA adapters.

    Attributes:
        top_k: Number of experts to select per token
        temperature: Softmax temperature for routing
        task_adapter_names: List of task-specific adapter names
        gks_adapter_names: List of general knowledge adapter names
        prototypes: Buffer holding prototype vectors [E, in_features]
    """

    def __init__(self, in_features, arrow_config):
        """Initialize Arrow routing layer."""

    def build_prototypes(self, lora_A, lora_B):
        """Compute prototype vectors via SVD for each adapter."""

    def gen_know_sub(self, lora_A, lora_B):
        """Perform General Knowledge Subtraction."""

    def forward(self, x, lora_A, lora_B, dropout, scaling):
        """Apply Arrow routing with top-k expert selection."""

def create_arrow_model(
    base_model: PreTrainedModel,
    task_specific_adapter_paths: list[str],
    arrow_config: ArrowConfig,
    general_adapter_paths: list[str] | None = None,
    **adapter_kwargs,
):
    """Create model with Arrow routing over loaded adapters."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft.tuners.lora.arrow import ArrowLoraLinearLayer, create_arrow_model, ArrowConfig
from peft import ArrowConfig
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| in_features || int || Yes || Input dimension for prototype computation
|-
| arrow_config || ArrowConfig || Yes || Configuration with top_k, temperature, adapter names
|-
| task_specific_adapter_paths || list[str] || Yes || Paths to task LoRA adapters
|-
| general_adapter_paths || list[str] || No || Paths to general knowledge adapters for GKS
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| forward output || torch.Tensor || Weighted combination of routed expert outputs
|-
| prototypes || torch.Tensor || Computed prototype vectors [num_experts, in_features]
|}

== Usage Examples ==

=== Creating Arrow Model ===
<syntaxhighlight lang="python">
from peft import ArrowConfig
from peft.tuners.lora.arrow import create_arrow_model
from transformers import AutoModelForCausalLM

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Configure Arrow
arrow_config = ArrowConfig(
    top_k=2,                    # Select top 2 experts per token
    router_temperature=0.1,     # Softmax temperature
    use_gks=False,              # Disable General Knowledge Subtraction
)

# Create Arrow model with multiple adapters
model = create_arrow_model(
    base_model=base_model,
    task_specific_adapter_paths=[
        "path/to/math_adapter",
        "path/to/code_adapter",
        "path/to/writing_adapter",
    ],
    arrow_config=arrow_config,
)
</syntaxhighlight>

=== Arrow with General Knowledge Subtraction ===
<syntaxhighlight lang="python">
from peft import ArrowConfig
from peft.tuners.lora.arrow import create_arrow_model

# Enable GKS to purify task-specific adapters
arrow_config = ArrowConfig(
    top_k=2,
    router_temperature=0.1,
    use_gks=True,               # Enable General Knowledge Subtraction
)

model = create_arrow_model(
    base_model=base_model,
    task_specific_adapter_paths=[
        "path/to/task1_adapter",
        "path/to/task2_adapter",
    ],
    general_adapter_paths=[
        "path/to/general_knowledge_adapter",
    ],
    arrow_config=arrow_config,
)
</syntaxhighlight>

=== Arrow Inference ===
<syntaxhighlight lang="python">
# Arrow automatically routes tokens to appropriate experts
outputs = model.generate(
    input_ids,
    max_new_tokens=100,
)

# Token routing happens internally:
# 1. Compute cosine similarity with prototypes
# 2. Select top-k experts per token
# 3. Combine expert outputs via weighted sum
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:huggingface_peft_Core_Environment]]
