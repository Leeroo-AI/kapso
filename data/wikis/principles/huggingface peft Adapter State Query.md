{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Doc|PEFT Documentation|https://huggingface.co/docs/peft]]
|-
! Domains
| [[domain::Multi_Task]], [[domain::Adapter]], [[domain::Model_Management]]
|-
! Last Updated
| [[last_updated::2025-01-15 12:00 GMT]]
|}

== Overview ==

Principle for inspecting the state of loaded adapters on a PEFT model, including active adapter identification and configuration retrieval.

=== Description ===

Adapter State Query provides introspection capabilities for multi-adapter PEFT models. This includes:
* Identifying which adapter(s) are currently active
* Retrieving configuration for all loaded adapters
* Getting detailed model status including adapter types and trainability

This is essential for debugging, logging, and conditional logic in multi-adapter serving scenarios.

=== Usage ===

Apply this principle when:
* Building multi-adapter serving systems that need runtime introspection
* Debugging adapter switching issues
* Logging which adapter was used for a given request
* Implementing conditional logic based on adapter state

== Theoretical Basis ==

'''Adapter Registry:'''

PEFT maintains internal registries tracking loaded adapters:

<syntaxhighlight lang="python">
class PeftModel:
    peft_config: dict[str, PeftConfig]  # name -> config mapping
    active_adapter: Union[str, list[str]]  # currently active adapter(s)
</syntaxhighlight>

'''State Properties:'''

* `model.peft_config` - Dictionary mapping adapter names to their configurations
* `model.active_adapter` - String or list of strings indicating active adapter(s)
* `model.get_model_status()` - Detailed breakdown of adapter types and states

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_peft_query_adapter_state]]
