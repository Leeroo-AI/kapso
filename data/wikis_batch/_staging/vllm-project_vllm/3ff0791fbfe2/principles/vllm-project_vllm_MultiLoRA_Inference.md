'''Metadata'''
{| class="wikitable"
|-
! Key !! Value
|-
| Knowledge Sources || vLLM Batching Architecture, Multi-Adapter Serving
|-
| Domains || Inference Optimization, Dynamic Batching
|-
| Last Updated || 2025-12-17
|}

== Overview ==

'''MultiLoRA Inference''' is the principle of executing inference requests with different LoRA adapters within the same computational batch, maximizing GPU utilization while maintaining adapter isolation and correctness. This enables efficient multi-tenant serving where diverse specialized adapters coexist on shared infrastructure.

== Description ==

The MultiLoRA Inference principle addresses the core challenge of serving multiple LoRA adapters efficiently: how to batch requests from different adapters to achieve high throughput without compromising per-adapter accuracy or creating memory bottlenecks.

=== Batching Strategy ===

Traditional inference batching assumes all requests use identical model weights. MultiLoRA inference extends this by:

* '''Adapter-Aware Scheduling''': The scheduler tracks which adapters are required for pending requests
* '''Dynamic Adapter Loading''': Loading adapters into GPU memory slots as requests arrive
* '''Mixed-Adapter Batching''': Computing forward passes where different sequences use different adapters
* '''Memory-Bounded Admission''': Limiting concurrent adapters based on max_loras configuration

When max_loras=1, the scheduler must batch only requests using the same adapter, potentially reducing batch sizes and GPU utilization. When max_loras>1, requests with different adapters can coexist in batches, improving throughput at the cost of increased memory consumption.

=== Computation Model ===

During mixed-adapter forward passes:

1. '''Base Model Computation''': All sequences process through shared base model layers identically
2. '''Adapter Selection''': Each sequence's adapter ID determines which LoRA weights to apply
3. '''Adapter Application''': LoRA transformations apply per-sequence using selected adapter weights
4. '''Result Aggregation''': Output sequences maintain their adapter associations through generation

The implementation uses efficient batched operations where base computations are shared and adapter-specific operations are parallelized across adapter slots.

=== Adapter Lifecycle ===

Adapters transition through states during serving:

* '''Cold''': Adapter on disk/CPU, not in GPU cache
* '''Loading''': Adapter weights transferring to GPU memory
* '''Active''': Adapter in GPU cache, available for batching
* '''Evicting''': Adapter being removed from GPU to free a slot

The scheduler coordinates these transitions, preferring to batch requests for already-loaded adapters to minimize cold-start overhead.

== Design Principles ==

* '''Transparent Batching''': From the request perspective, MultiLoRA inference is identical to single-adapter inference
* '''Isolation Guarantee''': One adapter's weights never affect another adapter's computations
* '''Fairness''': No adapter should be indefinitely starved due to higher-priority adapters
* '''Graceful Degradation''': When adapter count exceeds max_loras, system continues serving with time-slicing

== Performance Characteristics ==

* '''Best Case''': All requests use same adapter - equivalent to single-adapter performance
* '''Typical Case''': 2-4 adapters active - 70-90% of single-adapter throughput
* '''Worst Case''': Frequent adapter switching - 40-60% throughput due to loading overhead
* '''Memory Impact''': Each active adapter adds ~(rank × model_dim × 2 × num_layers) bytes

== Scheduling Strategies ==

* '''Adapter-First''': Prioritize batching requests from loaded adapters
* '''Fair Share''': Guarantee minimum throughput per adapter
* '''Priority-Based''': Weight adapters by SLA or tenant importance
* '''Preemptive Loading''': Predict upcoming adapters and preload to reduce cold starts

== Related Pages ==

* [[implemented_by::Implementation:vllm-project_vllm_LLMEngine_add_request]]
