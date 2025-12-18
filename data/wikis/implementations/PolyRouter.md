= PolyRouter =

== Knowledge Sources ==
* Source Repository: https://github.com/huggingface/peft
* Reference Implementation: https://github.com/microsoft/mttl/blob/ce4ca51dbca73be656feb9b3e5233633e3c5dec7/mttl/models/poly.py#L138
* Paper: [https://huggingface.co/papers/2202.13914 Polytropon (Poly)]
* Paper: [https://huggingface.co/papers/2211.03831 Multi-Head Routing (MHR)]

== Domains ==
* [[NLP]]
* [[PEFT]] (Parameter-Efficient Fine-Tuning)
* [[Multi-Task Learning]]
* [[Neural Network Routing]]
* [[Adaptive Computation]]

== Overview ==

=== Description ===
PolyRouter is a neural routing module that dynamically computes mixture weights for combining multiple LoRA "skills" in a Poly layer. It maintains learnable logits for each task and produces normalized weights using sigmoid activation and Gumbel-Softmax sampling during training.

The router implements task-specific routing by learning a matrix of logits (n_tasks × n_skills × n_splits) and computing normalized weights based on the task ID. During training, it uses RelaxedBernoulli (Gumbel-Softmax) sampling for differentiable routing; during inference, it uses deterministic sigmoid activation.

=== Usage ===
PolyRouter is instantiated automatically when creating Poly layers. It's called during forward passes with task IDs and input tensors to produce routing weights that determine how different LoRA skills are combined.

== Code Reference ==

=== Source Location ===
<code>src/peft/tuners/poly/router.py</code>

=== Signature ===
<syntaxhighlight lang="python">
class PolyRouter(Router):
    def __init__(self, poly_config: PolyConfig)
    def reset(self)
    def forward(self, task_ids: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor
</syntaxhighlight>

=== Import Statement ===
<syntaxhighlight lang="python">
from peft.tuners.poly.router import PolyRouter, get_router
</syntaxhighlight>

== I/O Contract ==

=== __init__ Method ===
{| class="wikitable"
! Parameter !! Type !! Description
|-
| poly_config || PolyConfig || Configuration containing n_tasks, n_skills, n_splits, and poly_type
|}

'''Creates:''' A PolyRouter with learnable module_logits parameter of shape (n_tasks, n_splits × n_skills)

=== forward Method ===
{| class="wikitable"
! Parameter !! Type !! Description
|-
| task_ids || torch.Tensor || Tensor containing task IDs (must not be None)
|-
| input_ids || torch.Tensor || Input tensor (currently unused in routing computation)
|}

'''Returns:''' torch.Tensor - Normalized routing weights of shape (batch_size, n_splits, n_skills)

'''Raises:'''
* ValueError if task_ids is None
* ValueError if any task_id >= n_tasks

=== reset Method ===
Reinitializes the module_logits parameter with uniform distribution U(-1e-3, 1e-3).

== Routing Computation ==

The forward pass computes routing weights through:

# '''Logit Selection''': <code>module_logits = self.module_logits[task_ids]</code>
# '''Reshape''': Reshape to (batch_size, n_splits, n_skills)
# '''Stochastic Sampling''' (training): <code>RelaxedBernoulli(temperature=1.0, logits=module_logits).rsample()</code>
# '''Deterministic''' (inference): <code>torch.sigmoid(module_logits)</code>
# '''Normalization''': <code>module_weights = module_logits / (module_logits.sum(dim=-1, keepdim=True) + EPS)</code>

Where EPS = 1e-12 prevents division by zero.

== Usage Examples ==

=== Creating a Router ===
<syntaxhighlight lang="python">
from peft.tuners.poly.config import PolyConfig
from peft.tuners.poly.router import get_router
import torch

# Create configuration
config = PolyConfig(
    r=8,
    n_tasks=5,
    n_skills=8,
    n_splits=2,
    poly_type="poly"
)

# Get router instance
router = get_router(config)

print(f"Router logits shape: {router.module_logits.shape}")
# Output: torch.Size([5, 16])  # 5 tasks, 16 = 2 splits × 8 skills
</syntaxhighlight>

=== Computing Routing Weights ===
<syntaxhighlight lang="python">
import torch
from peft.tuners.poly.router import PolyRouter
from peft.tuners.poly.config import PolyConfig

# Initialize router
config = PolyConfig(n_tasks=3, n_skills=4, n_splits=2)
router = PolyRouter(config)
router.eval()  # Set to inference mode

# Prepare task IDs
task_ids = torch.tensor([0, 1, 2])  # Batch of 3 samples with different tasks
input_ids = torch.randn(3, 128)  # Dummy input

# Compute routing weights
weights = router.forward(task_ids, input_ids)

print(f"Weights shape: {weights.shape}")
# Output: torch.Size([3, 2, 4])  # batch=3, splits=2, skills=4

print(f"Weights sum per split: {weights.sum(dim=-1)}")
# Each split's weights sum to approximately 1.0
</syntaxhighlight>

=== Training vs Inference Mode ===
<syntaxhighlight lang="python">
import torch
from peft.tuners.poly.router import PolyRouter
from peft.tuners.poly.config import PolyConfig

config = PolyConfig(n_tasks=2, n_skills=4, n_splits=1)
router = PolyRouter(config)

task_ids = torch.tensor([0])
input_ids = torch.randn(1, 128)

# Training mode: stochastic sampling
router.train()
weights_train_1 = router.forward(task_ids, input_ids)
weights_train_2 = router.forward(task_ids, input_ids)
print(f"Training mode - weights differ: {not torch.allclose(weights_train_1, weights_train_2)}")
# Output: True (sampling introduces randomness)

# Inference mode: deterministic
router.eval()
weights_eval_1 = router.forward(task_ids, input_ids)
weights_eval_2 = router.forward(task_ids, input_ids)
print(f"Eval mode - weights identical: {torch.allclose(weights_eval_1, weights_eval_2)}")
# Output: True (deterministic sigmoid)
</syntaxhighlight>

=== Resetting Router Weights ===
<syntaxhighlight lang="python">
import torch
from peft.tuners.poly.router import PolyRouter
from peft.tuners.poly.config import PolyConfig

config = PolyConfig(n_tasks=2, n_skills=4, n_splits=1)
router = PolyRouter(config)

# Original logits
original_logits = router.module_logits.clone()

# Modify logits
router.module_logits.data.fill_(10.0)

# Reset to small random values
router.reset()

print(f"After reset, logits are small: {router.module_logits.abs().max() < 1e-2}")
# Output: True
print(f"Logits range: [{router.module_logits.min():.6f}, {router.module_logits.max():.6f}]")
# Output approximately in [-0.001, 0.001]
</syntaxhighlight>

=== Multi-Head Routing Example ===
<syntaxhighlight lang="python">
import torch
from peft.tuners.poly.router import get_router
from peft.tuners.poly.config import PolyConfig

# Configure Multi-Head Routing with 4 splits
config = PolyConfig(
    n_tasks=10,
    n_skills=8,
    n_splits=4  # Multi-Head Routing
)

router = get_router(config)
router.eval()

# Process batch with different tasks
task_ids = torch.tensor([0, 2, 5, 9])
input_ids = torch.randn(4, 256)

weights = router.forward(task_ids, input_ids)

print(f"MHR weights shape: {weights.shape}")
# Output: torch.Size([4, 4, 8])  # batch=4, splits=4, skills=8

# Each split has independent routing
for split_idx in range(4):
    print(f"Split {split_idx} weights sum: {weights[:, split_idx, :].sum(dim=-1)}")
    # Each should sum to approximately 1.0
</syntaxhighlight>

== Implementation Details ==

=== Gumbel-Softmax Sampling ===
During training, RelaxedBernoulli (Gumbel-Softmax) provides:
* '''Differentiability''': Gradients flow through sampling operation
* '''Exploration''': Stochastic routing prevents collapse to single skill
* '''Temperature''': Fixed at 1.0 for balanced exploration/exploitation

=== Normalization ===
Weights are normalized by dividing by the sum across skills, ensuring each split's weights sum to 1.0. The epsilon constant (1e-12) prevents numerical issues.

=== Device Handling ===
Task IDs are automatically moved to the router's device: <code>task_ids = task_ids.to(self.module_logits.device)</code>

== Related Pages ==
* [[huggingface_peft_PolyConfig|PolyConfig]] - Configuration for Poly routing
* [[huggingface_peft_PolyModel|PolyModel]] - Model using PolyRouter
* [[huggingface_peft_PolyLayer|PolyLayer]] - Layer implementation
* [[Gumbel-Softmax]] - Sampling technique
* [[Multi-Task Learning]]
* [[Neural Network Routing]]

== Categories ==
[[Category:PEFT]]
[[Category:Routing]]
[[Category:Multi-Task Learning]]
[[Category:Neural Networks]]
[[Category:HuggingFace]]
