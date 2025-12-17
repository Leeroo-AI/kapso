# huggingface_peft_Adapter_Hotswapping

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Doc|PEFT Documentation|https://huggingface.co/docs/peft]]
|-
! Domains
| [[domain::LLMs]], [[domain::Inference]], [[domain::Production]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==

Enable zero-downtime adapter replacement for production deployments using PEFT's hot-swapping capabilities.

=== Description ===

This workflow covers PEFT's advanced hot-swapping feature that allows replacing adapter weights at runtime without reinitializing the model. This is critical for production systems that need to update models without service interruption, support per-request adapter selection, or dynamically load task-specific adapters.

**Key Benefits:**
* Zero-downtime adapter updates
* Sub-millisecond adapter switching
* Compatible with torch.compile() optimization
* Memory-efficient runtime adapter management

=== Usage ===

Execute this workflow when you need to:
* Update production model adapters without restart
* Serve multiple adapters with minimal latency overhead
* Build dynamic multi-tenant model serving systems
* Maintain torch.compile() optimization across adapter changes

**Prerequisites:**
* Production PEFT model deployment
* Adapters with compatible configurations (rank, target modules)
* Understanding of hot-swap constraints

== Execution Steps ==

=== Step 1: Prepare Model for Hot-Swapping ===
[[step::Principle:huggingface_peft_Hotswap_Preparation]]

Configure the model for hot-swap compatibility. This involves preparing internal data structures to support rapid weight replacement without graph recompilation.

'''Preparation steps:'''
* Convert scaling values to tensors for torch.compile compatibility
* Identify maximum rank across expected adapters
* Pre-pad adapter weights to accommodate different ranks
* Validate adapter configuration compatibility

=== Step 2: Configure Compiled Model (Optional) ===

For maximum performance, apply torch.compile() to the prepared model. The hot-swap mechanism is designed to work with compiled models without triggering recompilation.

'''Compilation setup:'''
* Call prepare_model_for_compiled_hotswap() first
* Then apply torch.compile() to the model
* Compiled graph remains valid across swaps
* Significant inference speedup retained

=== Step 3: Load Initial Adapter ===
[[step::Principle:huggingface_peft_Adapter_Loading]]

Load the first adapter that will serve requests. This adapter establishes the baseline configuration that subsequent hot-swapped adapters must match.

'''Initial load:'''
* Standard adapter loading via PeftModel
* Adapter configuration recorded for compatibility checking
* Model ready for serving
* First requests use this adapter

=== Step 4: Execute Hot-Swap ===
[[step::Principle:huggingface_peft_Hotswap_Execution]]

Replace the current adapter weights with a new adapter at runtime. The swap is atomic and does not require model reinitialization.

'''Hot-swap process:'''
* Load new adapter weights from checkpoint
* Validate configuration compatibility
* Copy new weights into existing adapter tensors
* Update scaling factors if needed
* Model immediately uses new adapter

=== Step 5: Handle Rank Mismatches ===

When swapping to adapters with different ranks, PEFT automatically handles padding/unpadding to maintain tensor shape compatibility.

'''Rank handling:'''
* Smaller rank: Zero-pad to match target shape
* Larger rank: Error if exceeds prepared maximum
* Automatic scaling adjustment
* Transparent to inference code

=== Step 6: Monitor and Validate ===

Verify that hot-swaps complete successfully and the model produces expected outputs with the new adapter.

'''Validation steps:'''
* Check adapter name reflects new adapter
* Verify scaling values updated correctly
* Run validation inference
* Monitor for any performance degradation

== Execution Diagram ==

{{#mermaid:graph TD
    A[Prepare for Hot-Swap] --> B{Use Compile?}
    B -->|Yes| C[torch.compile Model]
    B -->|No| D[Load Initial Adapter]
    C --> D
    D --> E[Serve Requests]
    E --> F{Swap Needed?}
    F -->|Yes| G[Execute Hot-Swap]
    G --> H[Validate Swap]
    H --> E
    F -->|No| E
}}

== Related Pages ==

* [[step::Principle:huggingface_peft_Hotswap_Preparation]]
* [[step::Principle:huggingface_peft_Adapter_Loading]]
* [[step::Principle:huggingface_peft_Hotswap_Execution]]
