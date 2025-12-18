{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Doc|LoRA Adapters|https://docs.vllm.ai/en/latest/models/lora.html]]
|-
! Domains
| [[domain::LLM_Inference]], [[domain::LoRA]], [[domain::Multi_Adapter]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
End-to-end process for serving multiple LoRA adapters dynamically on a single base model using vLLM's multi-LoRA capability.

=== Description ===
This workflow demonstrates vLLM's ability to serve multiple LoRA (Low-Rank Adaptation) fine-tuned models from a single base model. Instead of loading separate full models for each fine-tuned variant, vLLM loads the base model once and dynamically swaps lightweight LoRA adapters at inference time. This dramatically reduces memory usage and enables serving many specialized models concurrently.

=== Usage ===
Execute this workflow when you have multiple LoRA-fine-tuned variants of a base model (e.g., specialized for different tasks like SQL generation, code completion, or domain-specific knowledge) and want to serve them efficiently from shared GPU memory. This pattern is ideal for multi-tenant deployments or A/B testing different fine-tuned models.

== Execution Steps ==

=== Step 1: Engine Configuration with LoRA ===
[[step::Principle:vllm-project_vllm_LoRA_Engine_Configuration]]

Configure the vLLM engine with LoRA support enabled. Key parameters control the maximum number of concurrent LoRA adapters (`max_loras`), maximum adapter rank (`max_lora_rank`), and CPU cache size for adapter weights (`max_cpu_loras`). These settings determine memory allocation for adapter management.

'''Configuration parameters:'''
* `enable_lora=True` - Enable LoRA adapter support
* `max_loras` - Maximum concurrent adapters in GPU memory (affects batch scheduling)
* `max_lora_rank` - Maximum rank of LoRA adapters supported
* `max_cpu_loras` - Size of CPU LoRA cache for adapter swapping
* Base model loaded normally with all standard engine options

=== Step 2: LoRA Adapter Registration ===
[[step::Principle:vllm-project_vllm_LoRA_Adapter_Registration]]

Prepare LoRA adapter specifications using `LoRARequest` objects. Each adapter is identified by a unique name and integer ID, along with the path to the adapter weights (typically from HuggingFace Hub or local directory). Adapters are loaded on-demand when first requested.

'''LoRARequest structure:'''
* `lora_name` - Human-readable name for the adapter
* `lora_int_id` - Unique integer identifier for batching
* `lora_path` - Path to adapter weights (HF repo or local directory)
* Adapters can be dynamically added and removed at runtime

=== Step 3: Request Submission with Adapter ===
[[step::Principle:vllm-project_vllm_LoRA_Request_Submission]]

Submit inference requests with associated LoRA adapters. Each request can specify a different adapter (or no adapter for base model inference). Requests with the same adapter are batched together for efficient processing. The scheduler manages adapter swapping to maximize throughput.

'''Request submission:'''
* Pass `LoRARequest` to `engine.add_request()` or `llm.generate()`
* Requests without LoRA use the base model
* Same prompt can be run with different adapters for comparison
* Batching groups requests by adapter when possible

=== Step 4: Adapter-Aware Scheduling ===
[[step::Principle:vllm-project_vllm_LoRA_Scheduling]]

The vLLM scheduler manages adapter loading and batching. When `max_loras=1`, requests for different adapters are processed sequentially. With higher `max_loras`, multiple adapters can be active simultaneously. The scheduler optimizes for throughput while respecting memory constraints.

'''Scheduling behavior:'''
* Adapters loaded into GPU memory on first use
* LRU eviction policy for adapter cache management
* Requests grouped by adapter for efficient batching
* CPU-to-GPU adapter transfer overlapped with computation

=== Step 5: Output Processing ===
[[step::Principle:vllm-project_vllm_LoRA_Output_Processing]]

Process generation results as standard `RequestOutput` objects. The output includes information about which adapter was used for generation, enabling tracking and routing in multi-adapter deployments. Results can be compared across different adapters.

'''Output handling:'''
* Same output format as non-LoRA inference
* Adapter information tracked per request
* Multiple adapters can process same prompt for comparison
* Streaming supported with adapter-specific outputs

== Execution Diagram ==
{{#mermaid:graph TD
    A[Engine Configuration with LoRA] --> B[LoRA Adapter Registration]
    B --> C[Request Submission with Adapter]
    C --> D[Adapter-Aware Scheduling]
    D --> E[Output Processing]
    E -->|More Requests| C
}}

== Related Pages ==
* [[step::Principle:vllm-project_vllm_LoRA_Engine_Configuration]]
* [[step::Principle:vllm-project_vllm_LoRA_Adapter_Registration]]
* [[step::Principle:vllm-project_vllm_LoRA_Request_Submission]]
* [[step::Principle:vllm-project_vllm_LoRA_Scheduling]]
* [[step::Principle:vllm-project_vllm_LoRA_Output_Processing]]
