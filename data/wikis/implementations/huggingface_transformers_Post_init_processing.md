{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Model Loading Documentation|https://huggingface.co/docs/transformers/main_classes/model]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Model_Management]]
|-
! Last Updated
| [[last_updated::2025-12-18 00:00 GMT]]
|}

== Overview ==
Concrete tools for finalizing model state after weight loading through weight tying and initialization completion provided by HuggingFace Transformers.

=== Description ===
The tie_weights() and post_init() methods in PreTrainedModel handle the critical finalization steps after model loading. tie_weights() implements parameter sharing between specified layers (typically input embeddings and output projection layers in language models), reducing memory footprint and ensuring weight consistency. It handles complex scenarios including symmetric tying during checkpoint loading (where either source or target might be present) and supports regex-based patterns for tying multiple parameter groups. post_init() completes the model initialization by registering tied weight keys, collecting parallel execution plans from submodules, and setting up distributed inference configurations.

These methods are automatically called at the end of from_pretrained() but can also be invoked manually when implementing custom loading procedures or modifying model structure.

=== Usage ===
These methods are called automatically during model loading via from_pretrained(). Manual invocation is needed when: creating models from scratch, implementing custom model loading logic, modifying model architecture after loading, or debugging weight tying issues. They ensure models are in a valid state before inference or training.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' src/transformers/modeling_utils.py
* '''Lines:'''
  * post_init: 1354-1390
  * tie_weights: 2383-2450

=== Signature ===
<syntaxhighlight lang="python">
def post_init(self):
    """
    A method executed at the end of each Transformer model initialization,
    to execute code that needs the model's modules properly initialized
    (such as weight initialization).
    """

def tie_weights(
    self,
    missing_keys: Optional[set[str]] = None,
    recompute_mapping: bool = True
):
    """
    Tie the model weights. If recompute_mapping=False, it will rely on
    the model.all_tied_weights_keys attribute. If recompute_mapping=True,
    it will re-check all internal submodels and their config to determine
    the params that need to be tied.
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM
# Methods are called automatically, but can be invoked manually
</syntaxhighlight>

== I/O Contract ==

=== post_init() ===

==== Inputs ====
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| self || PreTrainedModel || Yes || The model instance to initialize
|}

==== Outputs ====
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| None || None || Modifies model in-place, sets up internal state
|}

=== tie_weights() ===

==== Inputs ====
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| self || PreTrainedModel || Yes || The model instance
|-
| missing_keys || set[str] || No || Set of parameter keys missing from checkpoint (used during loading)
|-
| recompute_mapping || bool || No || Whether to recompute tied weight mapping from config (default: True)
|}

==== Outputs ====
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| None || None || Modifies model parameters in-place, establishes weight sharing
|}

== Usage Examples ==

=== Automatic Weight Tying (Standard Loading) ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM

# tie_weights() and post_init() are called automatically
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Check if weights are tied
input_embeddings = model.transformer.wte.weight
output_embeddings = model.lm_head.weight

print(f"Input embeddings address: {input_embeddings.data_ptr()}")
print(f"Output embeddings address: {output_embeddings.data_ptr()}")
print(f"Weights are tied: {input_embeddings.data_ptr() == output_embeddings.data_ptr()}")
</syntaxhighlight>

=== Manual Weight Tying ===
<syntaxhighlight lang="python">
from transformers import GPT2LMHeadModel, GPT2Config

# Create model with config
config = GPT2Config()
model = GPT2LMHeadModel(config)

# Manually initialize weights (would normally happen in __init__)
model.apply(model._init_weights)

# Manually tie weights
model.tie_weights()

# Verify tying
print(f"Tied weights keys: {model.all_tied_weights_keys}")
</syntaxhighlight>

=== Inspecting Tied Weights ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")

# Get tied weight mappings
if hasattr(model, 'all_tied_weights_keys'):
    print("Tied weight mappings:")
    for target, source in model.all_tied_weights_keys.items():
        print(f"  {target} → {source}")

# Example output:
# lm_head.weight → transformer.wte.weight
</syntaxhighlight>

=== Memory Savings from Weight Tying ===
<syntaxhighlight lang="python">
from transformers import GPT2LMHeadModel, GPT2Config
import torch

config = GPT2Config(vocab_size=50257, n_embd=768)

# Model with tied weights (default)
model_tied = GPT2LMHeadModel(config)
model_tied.tie_weights()
memory_tied = sum(p.numel() * p.element_size() for p in model_tied.parameters()) / 1e6

# Manually untie for comparison (not recommended in practice)
model_untied = GPT2LMHeadModel(config)
model_untied.lm_head.weight = torch.nn.Parameter(
    model_untied.lm_head.weight.clone()
)
memory_untied = sum(p.numel() * p.element_size() for p in model_untied.parameters()) / 1e6

print(f"Memory with tied weights: {memory_tied:.2f} MB")
print(f"Memory without tying: {memory_untied:.2f} MB")
print(f"Memory saved: {memory_untied - memory_tied:.2f} MB")
print(f"Reduction: {(1 - memory_tied/memory_untied) * 100:.1f}%")
</syntaxhighlight>

=== Custom Post-Initialization ===
<syntaxhighlight lang="python">
from transformers import PreTrainedModel, GPT2Config
import torch.nn as nn

class CustomGPT2(PreTrainedModel):
    config_class = GPT2Config

    def __init__(self, config):
        super().__init__(config)
        # Build model architecture
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(config.n_embd, config.n_head),
            num_layers=config.n_layer
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

        # Call post_init to finalize
        self.post_init()

    def forward(self, input_ids):
        return self.lm_head(self.transformer(input_ids))

# post_init() is called automatically in __init__
config = GPT2Config()
model = CustomGPT2(config)
print("Model initialized with post_init() called")
</syntaxhighlight>

=== Debugging Weight Tying ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained("gpt2")

# Get embedding and output weights
wte = model.transformer.wte.weight
lm_head = model.lm_head.weight

print(f"wte shape: {wte.shape}")
print(f"lm_head shape: {lm_head.shape}")
print(f"Same data pointer: {wte.data_ptr() == lm_head.data_ptr()}")
print(f"Same storage: {wte.untyped_storage().data_ptr() == lm_head.untyped_storage().data_ptr()}")

# Modify one, check if other changes
original_value = wte[0, 0].item()
wte[0, 0] = 999.0
print(f"Modified wte[0,0] to 999.0")
print(f"lm_head[0,0] is now: {lm_head[0, 0].item()}")
print(f"Weights are truly tied: {lm_head[0, 0].item() == 999.0}")

# Restore
wte[0, 0] = original_value
</syntaxhighlight>

=== Handling Untied Models ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM

# Some models don't tie weights by design
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Check if model has tied weights config
if hasattr(model, '_tied_weights_keys') and model._tied_weights_keys:
    print(f"Model uses weight tying: {model._tied_weights_keys}")
else:
    print("Model does not use weight tying")

# Some architectures like T5 have separate encoder/decoder embeddings
</syntaxhighlight>

=== Recomputing Tied Weights Mapping ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")

# Get current mapping (cached)
current_mapping = model.all_tied_weights_keys.copy()
print(f"Current mapping: {current_mapping}")

# Recompute from config (in case config changed)
model.tie_weights(recompute_mapping=True)

# Get new mapping
new_mapping = model.all_tied_weights_keys
print(f"Recomputed mapping: {new_mapping}")
print(f"Mappings are same: {current_mapping == new_mapping}")
</syntaxhighlight>

=== Post-Init for Distributed Models ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    device_map="auto"
)

# post_init() sets up parallel execution plans
if hasattr(model, '_tp_plan'):
    print(f"Tensor parallelism plan: {model._tp_plan}")
if hasattr(model, '_pp_plan'):
    print(f"Pipeline parallelism plan: {model._pp_plan}")
if hasattr(model, '_ep_plan'):
    print(f"Expert parallelism plan: {model._ep_plan}")
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Post_Loading_Hooks]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_transformers_Loading_Environment]]
