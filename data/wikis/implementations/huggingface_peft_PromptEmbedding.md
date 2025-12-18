{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Paper|Prompt Tuning|https://arxiv.org/abs/2104.08691]]
|-
! Domains
| [[domain::NLP]], [[domain::PEFT]], [[domain::Prompt_Tuning]]
|-
! Last Updated
| [[last_updated::2025-12-18 14:00 GMT]]
|}

== Overview ==

Prompt embedding module that encodes virtual tokens into continuous prompt embeddings for prefix-based parameter-efficient fine-tuning.

=== Description ===

PromptEmbedding creates learnable virtual token embeddings that are prepended to the input sequence. The module supports three initialization strategies: random initialization, sampling from the model's vocabulary (SAMPLE_VOCAB), or initializing from a text prompt (TEXT). Text initialization tokenizes the provided string and uses those embeddings as starting points, which often leads to faster convergence than random initialization.

=== Usage ===

Use PromptEmbedding for prompt tuning where you want to learn soft prompts instead of modifying model weights. This is the most parameter-efficient method as only the prompt embeddings are trained. TEXT initialization is recommended when you have a good natural language description of the task.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft PEFT]
* '''File:''' [https://github.com/huggingface/peft/blob/main/src/peft/tuners/prompt_tuning/model.py src/peft/tuners/prompt_tuning/model.py]
* '''Lines:''' 1-106

=== Signature ===
<syntaxhighlight lang="python">
class PromptEmbedding(torch.nn.Module):
    """
    The model to encode virtual tokens into prompt embeddings.

    Args:
        config: PromptTuningConfig with num_virtual_tokens, token_dim, etc.
        word_embeddings: The word embeddings of the base transformer model

    Attributes:
        embedding: torch.nn.Embedding layer for virtual tokens

    Input Shape: (batch_size, total_virtual_tokens)
    Output Shape: (batch_size, total_virtual_tokens, token_dim)
    """

    def __init__(self, config, word_embeddings):
        """Initialize prompt embedding with optional text/vocab initialization."""

    def forward(self, indices):
        """Get embeddings for virtual token indices."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft import PromptEmbedding, PromptTuningConfig
from peft.tuners.prompt_tuning import PromptEmbedding
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| config || PromptTuningConfig || Yes || Configuration with num_virtual_tokens, token_dim
|-
| word_embeddings || nn.Module || Yes || Base model's word embedding layer
|-
| indices || torch.Tensor || Yes || Virtual token indices for forward pass
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| prompt_embeddings || torch.Tensor || Embeddings [batch, num_virtual_tokens, token_dim]
|}

== Usage Examples ==

=== Basic Prompt Tuning ===
<syntaxhighlight lang="python">
from peft import PromptTuningConfig, get_peft_model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")

# Random initialization
config = PromptTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=20,
)

model = get_peft_model(model, config)
model.print_trainable_parameters()
# Only prompt embeddings are trained
</syntaxhighlight>

=== Text-Initialized Prompt Tuning ===
<syntaxhighlight lang="python">
from peft import PromptTuningConfig, PromptTuningInit, get_peft_model

# Initialize prompts from text description
config = PromptTuningConfig(
    task_type="SEQ_2_SEQ_LM",
    num_virtual_tokens=20,
    prompt_tuning_init=PromptTuningInit.TEXT,
    prompt_tuning_init_text="Classify the sentiment of this review:",
    tokenizer_name_or_path="t5-base",
)

model = get_peft_model(model, config)
</syntaxhighlight>

=== Vocabulary-Sampled Initialization ===
<syntaxhighlight lang="python">
from peft import PromptTuningConfig, PromptTuningInit, get_peft_model

# Sample random tokens from vocabulary for initialization
config = PromptTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=10,
    prompt_tuning_init=PromptTuningInit.SAMPLE_VOCAB,
)

model = get_peft_model(model, config)
</syntaxhighlight>

=== Direct PromptEmbedding Usage ===
<syntaxhighlight lang="python">
from peft import PromptEmbedding, PromptTuningConfig

config = PromptTuningConfig(
    peft_type="PROMPT_TUNING",
    task_type="SEQ_2_SEQ_LM",
    num_virtual_tokens=20,
    token_dim=768,
    num_transformer_submodules=1,
    prompt_tuning_init="TEXT",
    prompt_tuning_init_text="Predict sentiment:",
    tokenizer_name_or_path="t5-base",
)

# Create prompt embedding using model's word embeddings
prompt_embedding = PromptEmbedding(config, t5_model.shared)

# Get embeddings for batch
indices = torch.arange(20).unsqueeze(0).expand(batch_size, -1)
embeddings = prompt_embedding(indices)
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:huggingface_peft_Core_Environment]]
