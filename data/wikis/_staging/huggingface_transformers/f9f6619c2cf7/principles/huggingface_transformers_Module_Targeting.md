{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|Transformers Quantization|https://huggingface.co/docs/transformers/main_classes/quantization]]
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2025-12-18 00:00 GMT]]
|}

== Overview ==
Selectively exclude specific modules from quantization to preserve numerical stability or maintain required precision for critical components.

=== Description ===
Module targeting addresses the practical reality that not all layers benefit equally from quantization. Output embedding layers, classification heads, and numerically sensitive modules often require full precision for model stability and accuracy. This principle establishes a mechanism for identifying and marking modules that should bypass quantization, whether through explicit user specification, model architecture conventions (e.g., "lm_head" in language models), or automatic detection of critical components like tied embeddings.

The challenge is balancing memory savings with accuracy preservation. Quantizing 99% of a model's parameters while keeping the final output layer in full precision typically retains nearly full accuracy while still providing massive memory reduction. The targeting system must handle hierarchical module names, support pattern matching, and merge user preferences with architecture-specific defaults.

=== Usage ===
Apply this principle when you need to:
* Keep output embedding layers in full precision for numerical stability
* Preserve classification heads that are sensitive to quantization errors
* Maintain full precision for fine-tuning adapters or LoRA layers
* Exclude modules that cause accuracy degradation when quantized
* Handle tied weights that should not be quantized independently
* Support user-specified skip lists for custom architectures
* Automatically detect architecture-specific modules requiring full precision

== Theoretical Basis ==

=== Module Identification Strategies ===

<pre>
1. Explicit user specification:
   skip_modules = ["lm_head", "model.embed_tokens"]
   # User knows specific modules to skip

2. Architectural defaults:
   # For CausalLM models, typically skip:
   - Last module in network (output head)
   - Output embedding layer
   - Any tied weight modules

3. Pattern-based matching:
   skip_patterns = ["*.norm", "*.layernorm"]
   # Skip all normalization layers

4. Type-based exclusion:
   skip_types = [nn.LayerNorm, nn.Embedding]
   # Skip all instances of certain module types
</pre>

=== Critical Module Detection ===

<pre>
Automatic detection algorithm:

def get_keys_to_not_convert(model):
    critical_modules = set()

    # 1. Tied weights (shared parameters)
    if model.all_tied_weights_keys:
        tied_keys = set(model.all_tied_weights_keys.values())
        critical_modules.update(tied_keys)

    # 2. Last module (often output projection)
    last_param_name = list(model.named_parameters())[-1][0]
    last_module = last_param_name.removesuffix(".weight")
    critical_modules.add(last_module)

    # 3. Output embedding layer
    output_emb = model.get_output_embeddings()
    if output_emb:
        output_emb_name = get_module_name(model, output_emb)
        critical_modules.add(output_emb_name)

    return list(critical_modules)

Example for GPT-2:
- Tied: "transformer.wte" (input embeddings) ↔ "lm_head" (output)
- Last: "lm_head"
- Output emb: "lm_head"
Result: {"lm_head", "transformer.wte"}
</pre>

=== Module Name Resolution ===

<pre>
Handle hierarchical module naming:

Full model structure:
model
├── model
│   ├── embed_tokens (Embedding)
│   ├── layers
│   │   ├── 0
│   │   │   ├── self_attn
│   │   │   │   ├── q_proj (Linear) ← Quantize this
│   │   │   │   ├── k_proj (Linear)
│   │   │   │   └── v_proj (Linear)
│   │   │   └── mlp
│   │   │       ├── gate_proj (Linear)
│   │   │       └── up_proj (Linear)
│   │   └── ...
│   └── norm (LayerNorm)
└── lm_head (Linear) ← Skip this

Module names from named_modules():
- "model.embed_tokens"
- "model.layers.0.self_attn.q_proj"
- "lm_head"

Skip list matching:
if "lm_head" in modules_to_not_convert:
    # Exact match: skip "lm_head"

if "model.layers.0" in modules_to_not_convert:
    # Prefix match: skip entire layer 0
</pre>

=== Merging Skip Lists ===

<pre>
Combine user preferences with defaults:

def get_modules_to_not_convert(model, skip_modules, add_default_skips):
    final_skip_list = []

    # Add defaults if requested
    if skip_modules is None or add_default_skips:
        default_skips = get_keys_to_not_convert(model)
        final_skip_list.extend(default_skips)

    # Add user-specified skips
    if skip_modules is not None:
        final_skip_list.extend(skip_modules)

    # Deduplicate
    return list(set(final_skip_list))

Example:
defaults = ["lm_head", "model.embed_tokens"]  # From architecture
user = ["model.layers.31.mlp.up_proj"]        # User wants to skip last layer MLP
final = ["lm_head", "model.embed_tokens", "model.layers.31.mlp.up_proj"]
</pre>

=== Skip List Application ===

<pre>
Use skip list during quantization:

def quantize_model(model, config, skip_modules):
    modules_to_skip = get_modules_to_not_convert(model, skip_modules, add_default_skips=True)

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if name in modules_to_skip:
                # Keep in full precision
                module.requires_grad = True  # If fine-tuning
                print(f"Skipping quantization: {name}")
            else:
                # Quantize module
                quantize_module(module, config)
                print(f"Quantized: {name}")

Result:
- lm_head: FP16 (skipped)
- model.embed_tokens: FP16 (skipped)
- model.layers.0.self_attn.q_proj: INT4 (quantized)
- model.layers.0.self_attn.k_proj: INT4 (quantized)
...
</pre>

=== Memory-Accuracy Trade-offs ===

<pre>
Quantization impact analysis:

Total model: 7B parameters
- Embeddings: 32K vocab × 4096 dim = 131M params (1.9%)
- Transformer layers: 6.6B params (94.3%)
- LM head: 32K vocab × 4096 dim = 131M params (1.9%)
- LayerNorms: ~130M params (1.9%)

Strategy 1: Quantize everything
- Memory: 7B × 0.5 bytes = 3.5GB
- Accuracy loss: Moderate

Strategy 2: Skip lm_head
- Memory: (7B - 131M) × 0.5 + 131M × 2 = 3.7GB
- Accuracy loss: Minimal (common practice)

Strategy 3: Skip lm_head + embeddings
- Memory: (7B - 262M) × 0.5 + 262M × 2 = 3.9GB
- Accuracy loss: Very minimal

Strategy 4: Skip lm_head + embeddings + LayerNorms
- Memory: (7B - 392M) × 0.5 + 392M × 2 = 4.1GB
- Accuracy loss: Negligible

Recommendation: Strategy 2 or 3 for best balance
</pre>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_transformers_Skip_modules_handling]]
