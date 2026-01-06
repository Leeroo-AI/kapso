{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|Transformers Model Loading|https://huggingface.co/docs/transformers/main_classes/model]]
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Model_Management]]
|-
! Last Updated
| [[last_updated::2025-12-18 00:00 GMT]]
|}

== Overview ==
Checkpoint Discovery is the systematic process of locating and resolving model weight files across different formats, storage locations, and sharding configurations.

=== Description ===
Modern deep learning models often have weights stored in multiple formats (safetensors, PyTorch pickle) and may be split across multiple files (sharded checkpoints). Checkpoint Discovery handles the complexity of determining which weight files to load based on format preferences, availability, and model configuration. It must handle local files, cached files, and remote repository files while respecting user preferences for safety (safetensors vs pickle), quantization formats (GGUF), and model variants (fp16, bf16).

The principle encompasses format negotiation, where the system attempts to load safer formats (safetensors) first before falling back to legacy formats (PyTorch pickle files). It also handles sharded models by discovering index files that map parameter names to their corresponding shard files, and manages automatic format conversion when appropriate.

=== Usage ===
Checkpoint Discovery should be applied after configuration resolution but before actual weight loading. Use it when you need to determine which files contain model weights, when working with models that might be sharded across multiple files, or when you want to prefer certain weight formats (like safetensors for security) over others.

== Theoretical Basis ==

The checkpoint discovery process implements a priority-based search strategy:

1. '''Format Priority''': Define preference order based on safety and performance
   * Priority: safetensors > PyTorch pickle > GGUF (if specified)
2. '''Variant Resolution''': Handle model variants (fp16, bf16, int8)
   * Weight filename = base_name + variant_suffix + extension
3. '''Sharding Detection''': Identify if model uses single or multiple weight files
   * Single: model.safetensors or pytorch_model.bin
   * Sharded: model.safetensors.index.json or pytorch_model.bin.index.json
4. '''Location Search''': Check multiple sources in order
   * Local directory
   * Local cache
   * Remote repository
5. '''Shard Expansion''': For sharded models, resolve all shard files from index

'''Search Algorithm''':
```
function discover_checkpoint_files(model_path, use_safetensors, variant):
    checkpoint_files = []

    # Priority 1: Explicit filename in config
    if config.has_explicit_filename():
        return resolve_file(config.get_explicit_filename())

    # Priority 2: Safetensors format
    if use_safetensors != False:
        # Try single file
        file = try_find(model_path, "model" + variant + ".safetensors")
        if file:
            return [file]

        # Try sharded safetensors
        index = try_find(model_path, "model" + variant + ".safetensors.index.json")
        if index:
            shards = parse_shard_index(index)
            return resolve_all_shards(shards)

    # Priority 3: PyTorch pickle format
    if use_safetensors != True:
        file = try_find(model_path, "pytorch_model" + variant + ".bin")
        if file:
            return [file]

        index = try_find(model_path, "pytorch_model" + variant + ".bin.index.json")
        if index:
            shards = parse_shard_index(index)
            return resolve_all_shards(shards)

    throw FileNotFoundError("No checkpoint files found")

function try_find(base_path, filename):
    # Check local first
    if is_local_directory(base_path):
        local_file = join_path(base_path, filename)
        if exists(local_file):
            return local_file

    # Check cache
    cached = check_cache(base_path, filename)
    if cached:
        return cached

    # Download from remote
    if is_remote_available():
        return download_and_cache(base_path, filename)

    return None
```

The invariants are:
* '''Completeness''': All weight files needed to reconstruct the full model must be discovered
* '''Consistency''': Format should be consistent across all discovered files (don't mix safetensors and pickle)
* '''Safety''': Prefer safer formats when available

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_transformers_Checkpoint_file_resolution]]
