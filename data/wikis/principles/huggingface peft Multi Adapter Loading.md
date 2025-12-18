{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Doc|PEFT Multi-Adapter|https://huggingface.co/docs/peft/conceptual_guides/lora#manage-multiple-adapters]]
|-
! Domains
| [[domain::Adapter]], [[domain::Multi_Task]], [[domain::Model_Loading]]
|-
! Last Updated
| [[last_updated::2025-01-15 12:00 GMT]]
|}

== Overview ==

Principle for loading multiple task-specific adapters into a single model for efficient multi-task inference.

=== Description ===

Multi-Adapter Loading enables a single base model to serve multiple tasks by:
1. Loading additional adapters after the initial PEFT setup
2. Storing multiple sets of adapter weights in memory
3. Enabling rapid switching between adapters
4. Sharing the base model parameters across all adapters

This is more memory-efficient than loading multiple full models.

=== Usage ===

Apply this when:
* One model needs to serve multiple tasks
* Switching between behaviors at inference time
* Memory efficiency is important (share base model)
* Tasks use compatible adapter configurations

== Theoretical Basis ==

'''Memory Efficiency:'''

Multi-adapter vs. multi-model memory:

Single model, N adapters:
<math>\text{Memory} = |W_0| + N \times |BA|</math>

N separate models:
<math>\text{Memory} = N \times |W_0|</math>

Savings ratio (for typical r=16, 7B model):
<math>\text{Savings} = \frac{N \times |W_0|}{|W_0| + N \times |BA|} \approx N</math>

'''Adapter Storage:'''

Each LoRA layer maintains a dictionary of adapters:
<syntaxhighlight lang="python">
# Pseudo-code for multi-adapter storage
class LoraLayer:
    def __init__(self):
        self.lora_A = {}  # adapter_name -> weight
        self.lora_B = {}  # adapter_name -> weight

    def add_adapter(self, name, r, alpha):
        self.lora_A[name] = nn.Parameter(torch.randn(r, in_features))
        self.lora_B[name] = nn.Parameter(torch.zeros(out_features, r))
</syntaxhighlight>

'''Forward Pass Selection:'''

During forward, only active adapter contributes:
<syntaxhighlight lang="python">
def forward(self, x, adapter_name):
    base_output = self.base_layer(x)

    if adapter_name in self.lora_A:
        lora_output = self.lora_B[adapter_name](self.lora_A[adapter_name](x))
        return base_output + lora_output * self.scaling

    return base_output
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_peft_load_adapter]]
