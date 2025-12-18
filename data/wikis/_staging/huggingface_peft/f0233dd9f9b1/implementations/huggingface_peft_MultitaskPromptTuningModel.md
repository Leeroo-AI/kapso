== MultitaskPromptEmbedding ==

=== Knowledge Sources ===
* [https://github.com/huggingface/peft PEFT Repository]
* [https://huggingface.co/papers/2303.02861 Multitask Prompt Tuning Paper]
* [https://mit-ibm-watson-ai-lab.github.io MIT-IBM Watson AI Lab]

=== Domains ===
[[Category:NLP]]
[[Category:PEFT]]
[[Category:Prompt_Tuning]]
[[Category:Multitask_Learning]]
[[Category:Neural_Networks]]

=== Overview ===

==== Description ====
'''MultitaskPromptEmbedding''' is a PyTorch neural network module that implements multitask prompt tuning embeddings. It extends the PromptEmbedding class to support multiple tasks simultaneously using a low-rank factorization approach.

The model represents task-specific prompts as the product of two low-rank matrices:
* '''prefix_task_cols''': Task-specific column matrix (num_tasks × total_virtual_tokens × num_ranks)
* '''prefix_task_rows''': Task-specific row matrix (num_tasks × num_ranks × token_dim)

This factorization allows efficient parameter sharing across tasks while maintaining task-specific adaptations. The architecture was developed at MIT-IBM Watson Research Lab as described in the paper [https://huggingface.co/papers/2303.02861 "Multitask Prompt Tuning Enables Parameter-Efficient Transfer Learning"].

==== Usage ====
Used as the embedding layer for multitask prompt tuning in PEFT models. It generates task-specific virtual token embeddings by combining shared prompt embeddings with task-specific transformations derived from low-rank matrices.

=== Code Reference ===

==== Source Location ====
<code>src/peft/tuners/multitask_prompt_tuning/model.py</code>

==== Signature ====
<syntaxhighlight lang="python">
class MultitaskPromptEmbedding(PromptEmbedding):
    def __init__(self, config: MultitaskPromptTuningConfig, word_embeddings):
        """
        Initialize multitask prompt embedding.

        Args:
            config: MultitaskPromptTuningConfig object
            word_embeddings: Base model word embeddings
        """

    def forward(self, indices, task_ids):
        """
        Generate task-specific prompt embeddings.

        Args:
            indices: Token indices for prompt
            task_ids: Task identifiers for batch samples

        Returns:
            Prompt embeddings modulated by task-specific factors
        """
</syntaxhighlight>

==== Import ====
<syntaxhighlight lang="python">
from peft.tuners.multitask_prompt_tuning import MultitaskPromptEmbedding
</syntaxhighlight>

=== I/O Contract ===

==== Constructor Parameters ====
{| class="wikitable"
! Parameter !! Type !! Description
|-
| config || MultitaskPromptTuningConfig || Configuration object specifying model parameters
|-
| word_embeddings || torch.nn.Embedding || Base model's word embedding layer
|}

==== Forward Method ====

===== Inputs =====
{| class="wikitable"
! Parameter !! Type !! Shape !! Description
|-
| indices || torch.Tensor || (batch_size, num_virtual_tokens) || Indices for prompt token embeddings
|-
| task_ids || torch.Tensor || (batch_size,) || Task identifiers for each sample in the batch
|}

===== Returns =====
{| class="wikitable"
! Type !! Shape !! Description
|-
| torch.Tensor || (batch_size, num_virtual_tokens, token_dim) || Task-specific prompt embeddings
|}

==== Model Attributes ====
{| class="wikitable"
! Attribute !! Type !! Description
|-
| num_tasks || int || Number of tasks supported
|-
| num_ranks || int || Rank of the low-rank factorization
|-
| num_virtual_tokens || int || Number of virtual tokens per task
|-
| num_transformer_submodules || int || Number of transformer submodules (1 for encoder-only, 2 for encoder-decoder)
|-
| token_dim || int || Dimension of token embeddings
|-
| prefix_task_cols || torch.nn.Parameter || Task-specific column matrix (learned)
|-
| prefix_task_rows || torch.nn.Parameter || Task-specific row matrix (learned)
|-
| embedding || torch.nn.Embedding || Shared prompt embedding layer (inherited)
|}

==== Side Effects ====
* Loads pretrained weights from state dict if using transfer learning initialization
* Creates trainable parameters for task-specific low-rank matrices

==== Exceptions ====
* '''ValueError''': Raised if task_ids is None in forward pass
* '''ValueError''': Raised if prompt_tuning_init_state_dict_path is None when using source task initialization

=== Usage Examples ===

==== Basic Usage ====
<syntaxhighlight lang="python">
import torch
from peft import MultitaskPromptTuningConfig
from peft.tuners.multitask_prompt_tuning import MultitaskPromptEmbedding

# Create configuration
config = MultitaskPromptTuningConfig(
    task_type="SEQ_2_SEQ_LM",
    num_virtual_tokens=20,
    token_dim=768,
    num_tasks=5,
    num_ranks=4,
    num_transformer_submodules=2
)

# Create word embeddings (from base model)
word_embeddings = torch.nn.Embedding(50000, 768)

# Initialize multitask prompt embedding
prompt_embedding = MultitaskPromptEmbedding(config, word_embeddings)

# Forward pass
batch_size = 8
indices = torch.arange(20).unsqueeze(0).expand(batch_size, -1)
task_ids = torch.randint(0, 5, (batch_size,))
embeddings = prompt_embedding(indices, task_ids)

print(embeddings.shape)  # torch.Size([8, 20, 768])
</syntaxhighlight>

==== Transfer Learning from Source Tasks ====
<syntaxhighlight lang="python">
import torch
from peft import MultitaskPromptTuningConfig
from peft.tuners.multitask_prompt_tuning import MultitaskPromptEmbedding

# Configuration with averaged source task initialization
config = MultitaskPromptTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=10,
    token_dim=768,
    num_tasks=3,
    num_ranks=2,
    prompt_tuning_init="AVERAGE_SOURCE_TASKS",
    prompt_tuning_init_state_dict_path="path/to/source_checkpoint.pt"
)

word_embeddings = torch.nn.Embedding(50000, 768)
prompt_embedding = MultitaskPromptEmbedding(config, word_embeddings)

# The model now has weights initialized from averaged source tasks
</syntaxhighlight>

==== Exact Source Task Initialization ====
<syntaxhighlight lang="python">
import torch
from peft import MultitaskPromptTuningConfig
from peft.tuners.multitask_prompt_tuning import MultitaskPromptEmbedding

# Initialize from specific source task
config = MultitaskPromptTuningConfig(
    task_type="SEQ_CLS",
    num_virtual_tokens=15,
    token_dim=1024,
    num_tasks=4,
    num_ranks=3,
    prompt_tuning_init="EXACT_SOURCE_TASK",
    prompt_tuning_init_state_dict_path="source_model.safetensors",
    prompt_tuning_init_task=1  # Use source task 1
)

word_embeddings = torch.nn.Embedding(50000, 1024)
prompt_embedding = MultitaskPromptEmbedding(config, word_embeddings)
</syntaxhighlight>

==== Understanding the Low-Rank Decomposition ====
<syntaxhighlight lang="python">
import torch
from peft import MultitaskPromptTuningConfig
from peft.tuners.multitask_prompt_tuning import MultitaskPromptEmbedding

config = MultitaskPromptTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=20,
    token_dim=768,
    num_tasks=3,
    num_ranks=4
)

word_embeddings = torch.nn.Embedding(50000, 768)
prompt_embedding = MultitaskPromptEmbedding(config, word_embeddings)

# The model creates low-rank matrices
print(f"Column matrix shape: {prompt_embedding.prefix_task_cols.shape}")
# Output: torch.Size([3, 20, 4])  # (num_tasks, total_virtual_tokens, num_ranks)

print(f"Row matrix shape: {prompt_embedding.prefix_task_rows.shape}")
# Output: torch.Size([3, 4, 768])  # (num_tasks, num_ranks, token_dim)

# During forward pass:
# task_prompts = matmul(task_cols, task_rows)
# This creates a (batch_size, total_virtual_tokens, token_dim) matrix
# that modulates the shared prompt embeddings
</syntaxhighlight>

=== Related Pages ===
* [[huggingface_peft_MultitaskPromptTuningConfig|MultitaskPromptTuningConfig]] - Configuration for this model
* [[huggingface_peft_PromptEmbedding|PromptEmbedding]] - Parent class
* [[huggingface_peft_PromptEncoder|PromptEncoder]] - Related P-tuning encoder
* [[PEFT|Parameter-Efficient Fine-Tuning]]
* [[Multitask_Learning|Multitask Learning]]
