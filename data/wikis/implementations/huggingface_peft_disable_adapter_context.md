{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
|-
! Domains
| [[domain::Adapter]], [[domain::Inference]], [[domain::Context_Manager]]
|-
! Last Updated
| [[last_updated::2025-01-15 12:00 GMT]]
|}

== Overview ==

Concrete tool for temporarily disabling all adapters to run inference on the base model.

=== Description ===

`disable_adapter` is a context manager that temporarily disables all adapter layers, allowing inference on the original base model without adapter effects. This is useful for comparing adapter vs. base model outputs or when base model behavior is needed temporarily.

=== Usage ===

Use this as a context manager with `with model.disable_adapter():`. Within the context, the model behaves as the original base model. Adapters are automatically re-enabled when exiting the context.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft peft]
* '''File:''' src/peft/peft_model.py
* '''Lines:''' L940-992

=== Signature ===
<syntaxhighlight lang="python">
@contextmanager
def disable_adapter(self):
    """
    Context manager to temporarily disable adapters.

    Usage:
        with model.disable_adapter():
            base_output = model(inputs)  # No adapter effect
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# Context manager on PeftModel
# with model.disable_adapter(): ...
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| (none) || - || - || Context manager takes no arguments
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| (context) || - || Within context, adapters are disabled. Re-enabled on exit.
|}

== Usage Examples ==

=== Compare Base vs Adapted Output ===
<syntaxhighlight lang="python">
# Get adapted output
model.set_adapter("math")
adapted_output = model.generate(**inputs)

# Get base model output
with model.disable_adapter():
    base_output = model.generate(**inputs)

# Compare outputs
print("Adapted:", tokenizer.decode(adapted_output[0]))
print("Base:", tokenizer.decode(base_output[0]))
</syntaxhighlight>

=== Selective Inference ===
<syntaxhighlight lang="python">
# Process some inputs with adapter, some without
for batch in dataloader:
    if batch["use_adapter"]:
        output = model(**batch["inputs"])
    else:
        with model.disable_adapter():
            output = model(**batch["inputs"])
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_peft_Adapter_Enable_Disable]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_peft_Core_Environment]]
