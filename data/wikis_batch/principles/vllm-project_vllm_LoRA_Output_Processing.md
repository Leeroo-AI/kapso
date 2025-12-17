'''Metadata'''
{| class="wikitable"
|-
! Key !! Value
|-
| Knowledge Sources || vLLM Output Processing, Request-Response Association
|-
| Domains || Response Generation, Adapter Traceability
|-
| Last Updated || 2025-12-17
|}

== Overview ==

'''LoRA Output Processing''' is the principle of maintaining adapter associations throughout the generation lifecycle and returning results with complete adapter context. This enables clients to trace which adapter produced each output and maintain audit trails for multi-adapter serving.

== Description ==

The LoRA Output Processing principle ensures that generated outputs preserve their connection to the LoRA adapter that produced them. In multi-adapter serving environments, this traceability is essential for debugging, monitoring, and ensuring clients receive correctly-attributed results.

=== Output Association ===

Throughout the generation process, the engine must maintain bidirectional mappings:

* '''Request → Adapter''': Which adapter is associated with each active request
* '''Output → Request''': Which request produced each generated token sequence
* '''Output → Adapter''': Transitive association from output to originating adapter

These associations persist through:
* Initial request submission
* Scheduling and batching operations
* Forward pass computation with adapter application
* Token generation and sampling
* Output assembly and streaming
* Final result delivery

=== Output Structure ===

LoRA-aware outputs include adapter metadata at multiple levels:

* '''RequestOutput.lora_request''': Top-level LoRARequest object for the entire request
* '''CompletionOutput.lora_request''': Adapter reference per completion (for n>1 sampling)
* Adapter information preserved in streaming outputs for incremental results
* Metrics and traces tagged with adapter identifiers

This structured association enables clients to:
* Verify correct adapter application
* Route outputs to appropriate downstream systems
* Aggregate metrics per adapter
* Debug adapter-specific generation issues

=== Streaming Considerations ===

In streaming mode, partial outputs must include adapter context:

* First chunk includes complete LoRARequest metadata
* Subsequent chunks maintain request_id for client-side association
* Final chunk confirms adapter used for full generation
* Streaming interruptions preserve adapter state for resumption

=== Multi-Output Scenarios ===

When generating multiple completions (n>1), output processing must:

* Apply the same adapter to all n samples consistently
* Include adapter reference in each CompletionOutput
* Aggregate metrics across all samples under single adapter
* Support fair comparison across samples with identical adapter

== Design Principles ==

* '''Immutable Association''': Once request-adapter binding established, it never changes
* '''Complete Metadata''': Every output includes sufficient information to identify its adapter
* '''Efficient Representation''': Adapter metadata references shared objects, not deep copies
* '''Backward Compatibility''': Outputs without LoRA (lora_request=None) remain valid

== Use Cases ==

* '''A/B Testing''': Comparing outputs from different adapters for same prompt
* '''Audit Compliance''': Tracing which model version produced regulated outputs
* '''Cost Attribution''': Billing different adapters/tenants for their usage
* '''Quality Monitoring''': Detecting degradation in specific adapter performance
* '''Debugging''': Isolating issues to particular adapters vs. base model

== Metadata Propagation ===

LoRA metadata flows through the system:

1. '''Input''': Client provides LoRARequest with add_request()
2. '''Scheduling''': Request queue entry includes adapter reference
3. '''Execution''': Batch metadata tracks per-sequence adapters
4. '''Sampling''': Generated tokens tagged with adapter context
5. '''Output Assembly''': RequestOutput constructed with LoRARequest
6. '''Delivery''': Client receives complete output with adapter metadata

== Related Pages ==

* [[implemented_by::Implementation:vllm-project_vllm_RequestOutput_lora]]
