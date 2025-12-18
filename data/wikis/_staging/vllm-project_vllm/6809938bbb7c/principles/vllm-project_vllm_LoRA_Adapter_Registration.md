{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|LoRA: Low-Rank Adaptation|https://arxiv.org/abs/2106.09685]]
* [[source::Doc|vLLM Documentation|https://docs.vllm.ai]]
|-
! Domains
| [[domain::NLP]], [[domain::LoRA]], [[domain::Model_Serving]]
|-
! Last Updated
| [[last_updated::2025-01-15 10:00 GMT]]
|}

== Overview ==

The process of defining and identifying LoRA adapters for use in inference requests, including naming, ID assignment, and path specification.

=== Description ===

LoRA Adapter Registration creates the binding between a logical adapter identity and its physical weights. This involves:

1. **Naming:** Human-readable identifier for the adapter
2. **ID Assignment:** Unique integer ID for internal tracking
3. **Path Resolution:** Locating adapter weights (HuggingFace or local)
4. **Validation:** Ensuring adapter is compatible with base model
5. **Metadata:** Optional settings like long-context support

Registration is the first step in using an adapter—it defines what adapter to use, but doesn't actually load it until needed.

=== Usage ===

Register LoRA adapters when:
- Setting up multi-adapter serving configurations
- Dynamically adding new adapters at runtime
- Building adapter selection logic for multi-tenant systems
- Organizing adapter collections for different use cases

== Theoretical Basis ==

'''Adapter Identity:'''

Each adapter needs unique identification:
- **Name:** For human readability and logging
- **Integer ID:** For efficient internal routing (must be > 0)
- **Path:** For locating weights

<syntaxhighlight lang="python">
# Adapter identification triple
adapter_identity = {
    "name": "sql-adapter",      # Human-readable
    "int_id": 1,                 # Internal routing key
    "path": "user/model-lora",  # Weight location
}
</syntaxhighlight>

'''Path Resolution:'''

vLLM supports multiple path formats:

<syntaxhighlight lang="python">
# HuggingFace Hub (downloads automatically)
lora_path = "yard1/llama-2-7b-sql-lora-test"

# Local path
lora_path = "/data/adapters/my_lora"

# S3-style (if configured)
lora_path = "s3://bucket/adapters/my_lora"
</syntaxhighlight>

'''Adapter Compatibility:'''

Adapters must match base model architecture:
- Same hidden dimensions
- Compatible attention configuration
- Matching tokenizer (or extra vocab configured)

<syntaxhighlight lang="python">
# Conceptual validation
def validate_adapter(adapter_path, base_model):
    adapter_config = load_adapter_config(adapter_path)

    # Check architecture match
    if adapter_config.base_model_name != base_model.name:
        warn("Adapter trained on different base model")

    # Check dimensions
    if adapter_config.hidden_size != base_model.hidden_size:
        raise ValueError("Hidden size mismatch")

    # Check rank
    if adapter_config.r > engine_args.max_lora_rank:
        raise ValueError(f"Adapter rank {adapter_config.r} > max {engine_args.max_lora_rank}")
</syntaxhighlight>

'''ID Management:'''

Integer IDs must be unique across all concurrent requests:

<syntaxhighlight lang="python">
# ID management pattern
class AdapterRegistry:
    def __init__(self):
        self.adapters = {}
        self.next_id = 1

    def register(self, name, path):
        if name in self.adapters:
            return self.adapters[name]

        adapter = LoRARequest(
            lora_name=name,
            lora_int_id=self.next_id,
            lora_path=path,
        )
        self.adapters[name] = adapter
        self.next_id += 1
        return adapter
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:vllm-project_vllm_LoRARequest_init]]
