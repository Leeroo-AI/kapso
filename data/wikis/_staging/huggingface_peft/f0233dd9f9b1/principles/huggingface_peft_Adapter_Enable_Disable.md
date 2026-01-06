{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
|-
! Domains
| [[domain::Adapter]], [[domain::Inference]], [[domain::Control]]
|-
! Last Updated
| [[last_updated::2025-01-15 12:00 GMT]]
|}

== Overview ==

Principle for temporarily disabling all adapter effects to run inference on the base model.

=== Description ===

Adapter Enable/Disable provides fine-grained control over adapter effects:
* **Disable:** All adapters bypass, only base model active
* **Enable:** Re-activate adapter contributions
* **Scoped:** Context manager ensures adapters re-enable on exit

This enables base model comparison without unloading adapters.

=== Usage ===

Apply this when:
* Comparing adapted vs. base model outputs
* Debugging adapter behavior
* Temporarily needing base model behavior
* A/B testing adapter effects

== Theoretical Basis ==

'''Disabled Forward Pass:'''

When adapters are disabled:
<math>h = W_0 x</math>

The LoRA contribution is bypassed entirely.

'''Implementation:'''

<syntaxhighlight lang="python">
# Pseudo-code for disable mechanism
class LoraLayer:
    def forward(self, x):
        result = self.base_layer(x)

        if not self.disable_adapters:  # Check flag
            for adapter_name in self.active_adapters:
                lora_out = self.lora_B[adapter_name](
                    self.lora_A[adapter_name](x)
                )
                result += lora_out * self.scaling[adapter_name]

        return result
</syntaxhighlight>

'''Context Manager Pattern:'''

The context manager ensures proper cleanup:
<syntaxhighlight lang="python">
@contextmanager
def disable_adapter(self):
    # Save current state
    old_state = self._adapters_disabled

    try:
        # Disable all adapters
        self._adapters_disabled = True
        for module in self.modules():
            if hasattr(module, 'disable_adapters'):
                module.disable_adapters = True
        yield
    finally:
        # Restore state (even if exception)
        self._adapters_disabled = old_state
        for module in self.modules():
            if hasattr(module, 'disable_adapters'):
                module.disable_adapters = old_state
</syntaxhighlight>

'''Use Cases:'''

1. **Baseline comparison:** Compare outputs before/after adaptation
2. **Debugging:** Isolate issues to adapter vs. base model
3. **Selective processing:** Some batches need base, some adapted

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_peft_disable_adapter_context]]
