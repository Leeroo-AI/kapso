# huggingface_peft_Multi_Adapter_Management

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Doc|PEFT Documentation|https://huggingface.co/docs/peft]]
|-
! Domains
| [[domain::LLMs]], [[domain::Fine_Tuning]], [[domain::Multi_Task]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==

Manage multiple PEFT adapters on a single base model for multi-task or multi-domain applications.

=== Description ===

This workflow enables sophisticated adapter management where a single base model hosts multiple specialized adapters. Each adapter can be independently loaded, activated, disabled, or merged. This pattern is powerful for multi-task learning, A/B testing, domain specialization, and adapter composition.

**Key Capabilities:**
* Load multiple adapters onto one base model
* Switch between adapters at runtime
* Combine adapters via weighted merging
* Use different adapter types (LoRA, IA3) together

=== Usage ===

Execute this workflow when you need to:
* Deploy one model serving multiple specialized tasks
* A/B test different fine-tuned versions
* Combine knowledge from multiple adapters
* Build modular AI systems with composable capabilities

**Prerequisites:**
* Base model with PEFT support
* Multiple trained adapter checkpoints
* Understanding of adapter compatibility constraints

== Execution Steps ==

=== Step 1: Load Base Model with First Adapter ===
[[step::Principle:huggingface_peft_Model_Loading]]

Initialize the PEFT model with the first adapter. This establishes the base model with one active adapter that subsequent adapters will join.

'''Initial setup:'''
* Load base model from pretrained
* Load first adapter via PeftModel.from_pretrained()
* First adapter becomes the active default
* Model ready to receive additional adapters

=== Step 2: Add Additional Adapters ===
[[step::Principle:huggingface_peft_Adapter_Addition]]

Load additional adapters onto the existing PEFT model. Each adapter gets a unique name for identification and switching.

'''Adding adapters:'''
* Use model.load_adapter() for each additional adapter
* Assign unique adapter_name to each
* Adapters can be from local paths or Hub
* Mixed adapter types supported (with limitations)

=== Step 3: Switch Active Adapter ===
[[step::Principle:huggingface_peft_Adapter_Switching]]

Change which adapter is currently active for inference. Only one adapter (or set of adapters) is active at a time during normal operation.

'''Switching mechanism:'''
* Use model.set_adapter(adapter_name) to switch
* Previous adapter deactivated (not unloaded)
* Switch is immediate with no model reload
* Can switch between any loaded adapters

=== Step 4: (Optional) Combine Multiple Adapters ===
[[step::Principle:huggingface_peft_Adapter_Combination]]

For supported adapter types (like LoRA), activate multiple adapters simultaneously or merge them with weighted averaging.

'''Combination options:'''
* Activate multiple: model.set_adapter([adapter1, adapter2])
* Weighted merge: Combine adapter weights mathematically
* TIES/DARE merging: Advanced merge algorithms
* Create new adapter from combination

=== Step 5: Merge Selected Adapters ===
[[step::Principle:huggingface_peft_Adapter_Merging]]

Merge one or more adapters into the base model weights permanently. This can be done selectively for deployment optimization.

'''Merge operations:'''
* merge_and_unload(): Merge active adapter into base
* Adapter weights added to base model
* Results in standard (non-PEFT) model
* Can save merged model for deployment

=== Step 6: Disable/Delete Adapters ===
[[step::Principle:huggingface_peft_Adapter_Lifecycle]]

Manage the lifecycle of loaded adapters by disabling or removing them as needed.

'''Lifecycle management:'''
* disable_adapter(): Temporarily disable all adapters
* delete_adapter(name): Remove adapter from memory
* Context manager for temporary disable
* Free memory from unused adapters

== Execution Diagram ==

{{#mermaid:graph TD
    A[Load Base + First Adapter] --> B[Add Adapter 2]
    B --> C[Add Adapter N]
    C --> D{Operations}
    D --> E[Switch Adapter]
    D --> F[Combine Adapters]
    D --> G[Merge Adapter]
    D --> H[Delete Adapter]
    E --> I[Inference]
    F --> I
    G --> J[Save Merged]
}}

== Related Pages ==

* [[step::Principle:huggingface_peft_Model_Loading]]
* [[step::Principle:huggingface_peft_Adapter_Addition]]
* [[step::Principle:huggingface_peft_Adapter_Switching]]
* [[step::Principle:huggingface_peft_Adapter_Combination]]
* [[step::Principle:huggingface_peft_Adapter_Merging]]
* [[step::Principle:huggingface_peft_Adapter_Lifecycle]]
