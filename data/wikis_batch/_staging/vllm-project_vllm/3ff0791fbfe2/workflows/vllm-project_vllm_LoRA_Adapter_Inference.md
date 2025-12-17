{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Doc|vLLM Documentation|https://docs.vllm.ai]]
* [[source::Paper|LoRA|https://arxiv.org/abs/2106.09685]]
|-
! Domains
| [[domain::LLMs]], [[domain::Fine_Tuning]], [[domain::Inference]], [[domain::LoRA]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==

End-to-end process for serving multiple LoRA (Low-Rank Adaptation) adapters with a shared base model, enabling efficient multi-task inference with minimal memory overhead.

=== Description ===

This workflow demonstrates vLLM's multi-LoRA serving capability, which allows running inference with multiple fine-tuned LoRA adapters on top of a single base model. Each request can specify a different LoRA adapter, enabling efficient multi-tenant or multi-task deployments without loading separate model instances.

Key capabilities:
* **Multi-LoRA serving**: Hot-swap between adapters per request
* **Memory efficiency**: Shared base model weights across adapters
* **Quantization support**: Combine LoRA with GPTQ, AWQ, bitsandbytes
* **Dynamic loading**: Load adapters on-demand from HuggingFace or local paths
* **Batch processing**: Mix requests for different adapters in same batch

=== Usage ===

Execute this workflow when you need to:
* Serve multiple fine-tuned model variants from a single deployment
* Support multi-tenant scenarios with user-specific adapters
* Run task-specific models (SQL, code, chat) with shared base
* Combine quantization with LoRA for memory-efficient fine-tuned inference

Ideal for scenarios where you have multiple LoRA adapters and want to serve them efficiently without dedicated GPU instances for each.

== Execution Steps ==

=== Step 1: Configure LoRA Engine Settings ===
[[step::Principle:vllm-project_vllm_LoRA_Engine_Configuration]]

Configure the engine with LoRA-specific parameters. This includes enabling LoRA support, setting maximum concurrent LoRAs, and configuring the maximum LoRA rank to support.

'''Key parameters:'''
* `enable_lora=True`: Enable LoRA adapter support
* `max_loras`: Maximum number of LoRAs loaded simultaneously (memory vs flexibility)
* `max_lora_rank`: Maximum rank of LoRA adapters to support
* `max_cpu_loras`: Number of LoRAs to cache on CPU for fast swapping

=== Step 2: Initialize Base Model Engine ===
[[step::Principle:vllm-project_vllm_LoRA_Base_Model_Loading]]

Load the base model with LoRA support enabled. The engine pre-allocates LoRA-specific buffers and sets up the adapter management infrastructure.

'''Initialization process:'''
1. Load base model weights to GPU
2. Allocate LoRA adapter slots based on `max_loras`
3. Initialize CPU LoRA cache for swapping
4. Set up adapter registry for request routing

=== Step 3: Load LoRA Adapters ===
[[step::Principle:vllm-project_vllm_LoRA_Adapter_Loading]]

Download and register LoRA adapters for use. Adapters can be loaded from HuggingFace Hub repositories or local paths. Each adapter receives a unique identifier for request routing.

'''Adapter sources:'''
* HuggingFace Hub repository IDs
* Local directory paths with adapter weights
* adapter_config.json specifies base model compatibility

=== Step 4: Create LoRA Requests ===
[[step::Principle:vllm-project_vllm_LoRA_Request_Creation]]

Create `LoRARequest` objects that associate prompts with specific adapters. Each request specifies the adapter name, unique ID, and path to the adapter weights.

'''LoRARequest structure:'''
* `lora_name`: Human-readable adapter name
* `lora_int_id`: Unique integer identifier for the adapter
* `lora_path`: Path to adapter weights (local or downloaded)

=== Step 5: Execute Multi-LoRA Inference ===
[[step::Principle:vllm-project_vllm_MultiLoRA_Inference]]

Submit requests with LoRA specifications for generation. The engine manages adapter loading/unloading, routes requests to appropriate adapters, and batches compatible requests.

'''Inference flow:'''
1. Requests queued with their LoRA specifications
2. Engine loads required adapters into active slots
3. Batch formed with compatible adapter configurations
4. Forward pass applies base model + adapter weights
5. Outputs returned with adapter attribution

=== Step 6: Process LoRA Results ===
[[step::Principle:vllm-project_vllm_LoRA_Output_Processing]]

Handle outputs from multi-LoRA inference. Results are structurally identical to standard inference but reflect the adapter-specific fine-tuning.

'''Output handling:'''
* Same `RequestOutput` structure as standard inference
* Results reflect adapter-specific behavior
* Can compare outputs across different adapters

== Execution Diagram ==
{{#mermaid:graph TD
    A[Configure LoRA Engine Settings] --> B[Initialize Base Model Engine]
    B --> C[Load LoRA Adapters]
    C --> D[Create LoRA Requests]
    D --> E[Execute Multi-LoRA Inference]
    E --> F[Process LoRA Results]
}}

== Related Pages ==
* [[step::Principle:vllm-project_vllm_LoRA_Engine_Configuration]]
* [[step::Principle:vllm-project_vllm_LoRA_Base_Model_Loading]]
* [[step::Principle:vllm-project_vllm_LoRA_Adapter_Loading]]
* [[step::Principle:vllm-project_vllm_LoRA_Request_Creation]]
* [[step::Principle:vllm-project_vllm_MultiLoRA_Inference]]
* [[step::Principle:vllm-project_vllm_LoRA_Output_Processing]]
