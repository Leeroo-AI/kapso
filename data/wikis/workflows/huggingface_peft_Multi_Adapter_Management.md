# Multi-Adapter Management

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Doc|PEFT Documentation|https://huggingface.co/docs/peft]]
|-
! Domains
| [[domain::LLMs]], [[domain::Multi_Task]], [[domain::Model_Serving]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==

End-to-end process for managing multiple PEFT adapters on a single base model, enabling dynamic task switching and efficient multi-task inference without reloading models.

=== Description ===

This workflow covers loading, switching, enabling/disabling, and managing multiple adapters on a single PEFT model. Multiple adapters can coexist in memory, and you can switch between them at runtime without reloading the base model. This is useful for serving multiple task-specific models from a single deployment.

=== Usage ===

Execute this workflow when:
* You need to serve multiple tasks from a single model deployment
* You want to A/B test different adapters
* You're building a multi-tenant system with per-user adapters
* You need to dynamically switch model behavior at runtime

== Execution Steps ==

=== Step 1: Load Base Model with First Adapter ===
[[step::Principle:huggingface_peft_Adapter_Loading]]

Load the base model and first adapter using `PeftModel.from_pretrained()`. This establishes the PEFT model structure with the first adapter as "default" (or with a custom name).

'''Initial setup:'''
* Load base model and first adapter
* Optionally name the adapter for clarity
* First adapter is automatically set as active

=== Step 2: Load Additional Adapters ===
[[step::Principle:huggingface_peft_Multi_Adapter_Loading]]

Load additional adapters using `model.load_adapter()`. Each adapter receives a unique name for identification and switching.

'''Loading pattern:'''
* `model.load_adapter("path/to/adapter2", adapter_name="task2")`
* Adapters share base model weights in memory
* Each adapter adds only its small weights (~MB) to memory

=== Step 3: Switch Active Adapter ===
[[step::Principle:huggingface_peft_Adapter_Switching]]

Switch between loaded adapters using `model.set_adapter()`. The active adapter determines which adapter weights are applied during forward passes.

'''Switching:'''
* `model.set_adapter("task2")` activates the "task2" adapter
* Instant switch, no model reloading
* Previous adapter remains loaded (can switch back)
* Check active adapter: `model.active_adapter`

=== Step 4: Disable/Enable Adapters ===
[[step::Principle:huggingface_peft_Adapter_Enable_Disable]]

Temporarily disable all adapters to run inference with the pure base model, then re-enable for adapter-augmented inference.

'''Disable pattern:'''
* Use context manager: `with model.disable_adapter(): ...`
* Or explicit: `model.disable_adapters()` / `model.enable_adapters()`
* Useful for comparing base vs. adapted model outputs

=== Step 5: Delete Unused Adapters ===
[[step::Principle:huggingface_peft_Adapter_Deletion]]

Remove adapters that are no longer needed to free memory. This permanently removes the adapter weights from the model.

'''Cleanup:'''
* `model.delete_adapter("adapter_name")` removes the adapter
* Cannot be undone - must reload if needed again
* Useful for managing memory in long-running processes

=== Step 6: Query Adapter State ===
[[step::Principle:huggingface_peft_Adapter_State_Query]]

Check the current state of loaded adapters, which is active, and their configurations.

'''State inspection:'''
* `model.peft_config` - dict of all loaded adapter configs
* `model.active_adapter` - currently active adapter name
* `model.get_model_status()` - detailed adapter status

== Execution Diagram ==

{{#mermaid:graph TD
    A[Load First Adapter] --> B[Load Additional Adapters]
    B --> C{Select Operation}
    C -->|Switch| D[Set Active Adapter]
    C -->|Compare| E[Disable/Enable Adapter]
    C -->|Cleanup| F[Delete Adapter]
    C -->|Inspect| G[Query State]
    D --> H[Run Inference]
    E --> H
    G --> C
    F --> C
}}

== Related Pages ==

* [[step::Principle:huggingface_peft_Adapter_Loading]]
* [[step::Principle:huggingface_peft_Multi_Adapter_Loading]]
* [[step::Principle:huggingface_peft_Adapter_Switching]]
* [[step::Principle:huggingface_peft_Adapter_Enable_Disable]]
* [[step::Principle:huggingface_peft_Adapter_Deletion]]
* [[step::Principle:huggingface_peft_Adapter_State_Query]]
