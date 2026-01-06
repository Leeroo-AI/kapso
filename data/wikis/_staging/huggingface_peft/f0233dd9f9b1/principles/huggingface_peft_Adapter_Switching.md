{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
|-
! Domains
| [[domain::Adapter]], [[domain::Multi_Task]], [[domain::Inference]]
|-
! Last Updated
| [[last_updated::2025-01-15 12:00 GMT]]
|}

== Overview ==

Principle for dynamically selecting which adapter(s) affect model outputs during inference.

=== Description ===

Adapter Switching enables rapid task switching by selecting which loaded adapter is "active" during forward passes. Only active adapters contribute to the model output. This enables:
* Single-adapter mode: One adapter active
* Multi-adapter mode: Multiple adapters sum their contributions
* No-adapter mode: Base model only (via disable_adapter)

=== Usage ===

Apply this when:
* Different inputs require different adaptations
* Comparing outputs across adapters
* Dynamically selecting task behavior
* Combining multiple adapter effects

== Theoretical Basis ==

'''Single Adapter Selection:'''

With adapter <math>i</math> selected:
<math>h = W_0 x + \frac{\alpha_i}{r_i} B_i A_i x</math>

Other adapters are present but don't contribute to output.

'''Multi-Adapter Combination:'''

When multiple adapters are active:
<math>h = W_0 x + \sum_{i \in \text{active}} \frac{\alpha_i}{r_i} B_i A_i x</math>

Contributions are summed (not averaged).

'''Implementation:'''

<syntaxhighlight lang="python">
# Pseudo-code for adapter selection
class LoraLayer:
    def forward(self, x):
        result = self.base_layer(x)

        # Only active adapters contribute
        for adapter_name in self.active_adapters:
            if adapter_name in self.lora_A:
                lora_out = self.lora_B[adapter_name](
                    self.lora_A[adapter_name](x)
                )
                result += lora_out * self.scaling[adapter_name]

        return result
</syntaxhighlight>

'''Switching Cost:'''

Adapter switching is effectively zero-cost:
* No weight copies
* No recomputation
* Just pointer/index update

<syntaxhighlight lang="python">
def set_adapter(self, adapter_name):
    # O(1) operation - just update active list
    self.active_adapter = adapter_name
    for module in self.modules():
        if hasattr(module, 'active_adapter'):
            module.active_adapter = adapter_name
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_peft_set_adapter]]
