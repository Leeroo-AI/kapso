{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Doc|PEFT Docs|https://huggingface.co/docs/peft]]
|-
! Domains
| [[domain::Multi_Task]], [[domain::Adapter]], [[domain::Model_Management]]
|-
! Last Updated
| [[last_updated::2025-01-15 12:00 GMT]]
|}

== Overview ==

Concrete tool for querying the state of loaded adapters on a PEFT model, providing introspection for active adapters and configurations.

=== Description ===

This implementation covers the properties and methods for inspecting PEFT model adapter state:
* `model.active_adapter` - Property returning the currently active adapter name(s)
* `model.peft_config` - Dictionary of all loaded adapter configurations
* `model.get_model_status()` - Detailed status report of all adapters

=== Usage ===

Use these properties to inspect adapter state during multi-adapter serving, debugging, or logging scenarios.

== Code Reference ==

=== Source Location ===
* '''File:''' `src/peft/peft_model.py`
* '''Lines:''' L180-250

=== Properties ===
<syntaxhighlight lang="python">
@property
def active_adapter(self) -> Union[str, list[str]]:
    """Return the name(s) of the currently active adapter(s)."""
    return self.base_model.active_adapter

@property
def peft_config(self) -> dict[str, PeftConfig]:
    """Return dictionary mapping adapter names to their configurations."""
    return self._peft_config

def get_model_status(self) -> PeftModelStatus:
    """Get detailed status of all adapters including type, trainability, etc."""
    ...
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft import PeftModel
# Properties accessed on PeftModel instance
</syntaxhighlight>

== Usage Examples ==

=== Inspecting Active Adapter ===
<syntaxhighlight lang="python">
from peft import PeftModel

# Load model with multiple adapters
model = PeftModel.from_pretrained(base_model, "adapter1")
model.load_adapter("adapter2", adapter_name="task2")

# Check active adapter
print(model.active_adapter)  # "default"

# Switch and check
model.set_adapter("task2")
print(model.active_adapter)  # "task2"
</syntaxhighlight>

=== Listing All Adapters ===
<syntaxhighlight lang="python">
# Get all loaded adapter configurations
for name, config in model.peft_config.items():
    print(f"Adapter '{name}': r={config.r}, target_modules={config.target_modules}")
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_peft_Adapter_State_Query]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_peft_Core_Environment]]
