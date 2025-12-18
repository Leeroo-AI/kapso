# Adapter Merging

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Doc|PEFT Documentation|https://huggingface.co/docs/peft]]
* [[source::Paper|TIES-Merging Paper|https://arxiv.org/abs/2306.01708]]
* [[source::Paper|DARE Paper|https://arxiv.org/abs/2311.03099]]
|-
! Domains
| [[domain::LLMs]], [[domain::Model_Merging]], [[domain::Multi_Task]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==

End-to-end process for combining multiple trained PEFT adapters into a single adapter using advanced merging algorithms (TIES, DARE, task arithmetic), enabling multi-task models without separate adapter switching.

=== Description ===

This workflow covers merging multiple task-specific LoRA adapters into a unified adapter that combines their capabilities. PEFT provides several merging algorithms: simple weighted averaging (task arithmetic), TIES (resolving sign conflicts), DARE (random pruning with rescaling), and combinations thereof. Merged adapters can perform multiple tasks without runtime switching overhead.

=== Usage ===

Execute this workflow when:
* You have multiple task-specific adapters and want a single multi-task adapter
* You want to combine domain knowledge from different adapters
* You need to reduce deployment complexity by eliminating adapter switching
* You're experimenting with model merging techniques

== Execution Steps ==

=== Step 1: Load Base Model ===
[[step::Principle:huggingface_peft_Base_Model_Loading]]

Load the base model that all adapters were trained from. All adapters being merged must share the same base model architecture.

'''Requirements:'''
* Same base model used for all adapters being merged
* Adapters must have compatible configurations (same target modules)

=== Step 2: Load Primary Adapter ===
[[step::Principle:huggingface_peft_Adapter_Loading]]

Load the first adapter using `PeftModel.from_pretrained()`. This creates the PEFT model structure that will hold the merged result.

'''Purpose:'''
* Establishes the PEFT model structure
* First adapter becomes the "default" adapter slot
* Configuration defines the structure for merging

=== Step 3: Load Additional Adapters ===
[[step::Principle:huggingface_peft_Multi_Adapter_Loading]]

Load additional adapters using `model.load_adapter()`. Each adapter is loaded with a unique name for identification during merging.

'''Loading multiple adapters:'''
* Use `load_adapter(path, adapter_name="unique_name")` for each
* All adapters coexist in memory with separate names
* Verify compatibility: same rank, target modules, and base model

=== Step 4: Configure Merge Strategy ===
[[step::Principle:huggingface_peft_Merge_Strategy_Configuration]]

Select and configure the merging algorithm. Different algorithms have different strengths:

'''Merging algorithms:'''
* **Linear/Task Arithmetic**: Weighted sum of adapter deltas
* **TIES**: Trims small values, elects majority sign, averages disjoint sets
* **DARE Linear**: Random pruning with rescaling + linear merge
* **DARE TIES**: Random pruning + TIES merging

'''Parameters:'''
* `weights`: List of weights for each adapter (importance)
* `density`: Fraction of values to keep (for TIES/DARE)
* `majority_sign_method`: "total" or "frequency" for sign election

=== Step 5: Execute Adapter Merge ===
[[step::Principle:huggingface_peft_Adapter_Merge_Execution]]

Call `model.add_weighted_adapter()` to merge the loaded adapters into a new combined adapter. The merging algorithm computes the weighted combination of adapter weights.

'''Merge execution:'''
* Specify adapters to merge by name
* Provide weights for each adapter
* New adapter is created with the combined weights
* Original adapters remain intact (can be deleted if needed)

=== Step 6: Evaluate Merged Adapter ===
[[step::Principle:huggingface_peft_Merge_Evaluation]]

Test the merged adapter on tasks from each source adapter to verify capability preservation. Merging may cause degradation on individual tasks.

'''Evaluation:'''
* Test on each source task to measure capability retention
* Compare to individual adapter performance
* Adjust weights or merging algorithm if needed

=== Step 7: Save Merged Adapter ===
[[step::Principle:huggingface_peft_Adapter_Serialization]]

Save the merged adapter for deployment. The merged adapter can be loaded just like any single adapter.

'''Output:'''
* Single adapter checkpoint with combined capabilities
* Same loading procedure as individual adapters
* Optionally delete source adapters after merge

== Execution Diagram ==

{{#mermaid:graph TD
    A[Load Base Model] --> B[Load Primary Adapter]
    B --> C[Load Additional Adapters]
    C --> D[Configure Merge Strategy]
    D --> E[Execute Merge]
    E --> F[Evaluate Merged Adapter]
    F --> G{Performance OK?}
    G -->|Yes| H[Save Merged Adapter]
    G -->|No| D
}}

== Related Pages ==

* [[step::Principle:huggingface_peft_Base_Model_Loading]]
* [[step::Principle:huggingface_peft_Adapter_Loading]]
* [[step::Principle:huggingface_peft_Multi_Adapter_Loading]]
* [[step::Principle:huggingface_peft_Merge_Strategy_Configuration]]
* [[step::Principle:huggingface_peft_Adapter_Merge_Execution]]
* [[step::Principle:huggingface_peft_Merge_Evaluation]]
* [[step::Principle:huggingface_peft_Adapter_Serialization]]
