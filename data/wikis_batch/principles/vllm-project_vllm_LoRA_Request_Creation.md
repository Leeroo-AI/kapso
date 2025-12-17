'''Metadata'''
{| class="wikitable"
|-
! Key !! Value
|-
| Knowledge Sources || vLLM LoRA Request API, Multi-Adapter Serving
|-
| Domains || Request Routing, Adapter Identification
|-
| Last Updated || 2025-12-17
|}

== Overview ==

'''LoRA Request Creation''' is the principle of associating inference requests with specific LoRA adapters through structured request metadata. This enables the serving system to route requests to appropriate adapter instances and maintain adapter-specific generation context.

== Description ==

The LoRA Request Creation principle defines how clients specify which adapter should be applied to a particular generation request. Since a LoRA-enabled engine can serve multiple adapters concurrently, each request must carry unambiguous adapter identification that the scheduler uses for batching and resource allocation decisions.

=== Request Components ===

A LoRA request specification contains three essential elements:

1. '''Adapter Name''': Human-readable identifier for logging, monitoring, and debugging
2. '''Adapter ID''': Unique integer identifier used by the engine for internal routing and cache management
3. '''Adapter Path''': Filesystem or repository location containing the adapter weights

The adapter ID (lora_int_id) serves as the primary routing key. Requests with the same ID are batched together and executed with the same adapter weights. IDs must be globally unique across all adapters served by the engine - reusing an ID for different adapters causes undefined behavior.

=== Request Lifecycle ===

LoRA requests flow through the serving pipeline maintaining their adapter association:

* '''Submission''': Client creates LoRARequest object and attaches to generation parameters
* '''Validation''': Engine verifies adapter ID uniqueness and path accessibility
* '''Loading''': If adapter not in cache, engine loads weights from specified path
* '''Batching''': Scheduler groups requests by adapter ID for efficient execution
* '''Execution''': Model applies specified adapter transformations during forward pass
* '''Output''': Results include LoRARequest reference for client-side routing

=== Multi-Adapter Coordination ===

When max_loras > 1, the engine can batch requests from different adapters in the same forward pass. The batching scheduler considers:

* '''Active Adapter Count''': Current number of loaded adapters vs. max_loras limit
* '''Memory Availability''': GPU memory for additional adapter weights
* '''Request Priority''': Fairness and latency objectives across adapters
* '''Cache State''': Preference for adapters already in GPU cache

== Design Principles ==

* '''Explicit Association''': Adapters never applied implicitly - requests without LoRARequest use base model
* '''ID Immutability''': Once assigned, adapter IDs should not change during engine lifetime
* '''Path Flexibility''': Same adapter can be loaded from different paths (local, HF Hub, cloud)
* '''Optional Usage''': LoRA-enabled engines support both LoRA and base model requests

== Adapter Identification Strategies ==

* '''Sequential IDs''': 1, 2, 3, ... for simple scenarios
* '''Hash-Based IDs''': Derived from adapter path for deterministic identification
* '''User-Assigned IDs''': Client-specified for multi-tenant routing
* '''Semantic IDs''': Encode adapter properties (task, domain, version) in ID space

== Operational Considerations ==

* '''ID Collision Prevention''': Centralized ID registry in multi-engine deployments
* '''Adapter Discovery''': Registry service mapping names to paths and IDs
* '''Version Management''': Incorporating version info in adapter names
* '''Monitoring''': Tracking per-adapter metrics (requests, latency, throughput)

== Related Pages ==

* [[implemented_by::Implementation:vllm-project_vllm_LoRARequest]]
