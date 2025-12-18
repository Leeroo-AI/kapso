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
State Dict Loading is the process of deserializing model weights from checkpoint files into memory as a dictionary mapping parameter names to tensor values.

=== Description ===
Once checkpoint files have been discovered and the model architecture has been instantiated, the actual weight values must be loaded from disk into memory. State Dict Loading handles the deserialization of different checkpoint formats (safetensors, PyTorch pickle files) into a standardized dictionary structure where keys are parameter names (like "encoder.layer.0.weight") and values are the corresponding tensor weights. This process must handle different serialization formats, manage memory efficiently (particularly for large models), support meta-device tensors for lazy loading, and ensure that loaded tensors are placed on appropriate devices.

The principle abstracts the differences between safetensors (which uses memory-mapped files for safety and efficiency) and PyTorch's native serialization format (torch.save/torch.load), providing a unified interface for weight loading regardless of the underlying format.

=== Usage ===
State Dict Loading should be applied after checkpoint file discovery and model architecture instantiation, but before the weights are actually assigned to model parameters. Use this when you need to inspect weight values, implement custom weight loading logic, or when working with sharded models where weights must be loaded incrementally.

== Theoretical Basis ==

State dictionary loading implements format-specific deserialization:

1. '''Format Detection''': Determine checkpoint file format from extension
   * .safetensors → Use safetensors library
   * .bin, .pt, .pth → Use PyTorch torch.load
2. '''Deserialization Strategy''':
   * '''Safetensors''': Memory-mapped loading with zero-copy reads
   * '''PyTorch Pickle''': Full deserialization with pickle protocol
3. '''Device Placement''': Control where tensors are loaded
   * "cpu": Load to CPU memory
   * "cuda:0": Load directly to GPU
   * "meta": Create tensor metadata without allocating memory
4. '''Safety Validation''': For PyTorch format, optionally validate for malicious code
5. '''Tensor Extraction''': Create {name: tensor} dictionary

'''Loading Algorithm''':
```
function load_state_dict(checkpoint_file, map_location="cpu", weights_only=True):
    # Detect format
    if checkpoint_file.endswith(".safetensors"):
        return load_safetensors(checkpoint_file, map_location)
    else:
        return load_pytorch_checkpoint(checkpoint_file, map_location, weights_only)

function load_safetensors(checkpoint_file, map_location):
    state_dict = {}

    # Open with memory mapping (efficient, no full load into memory)
    with safe_open(checkpoint_file, framework="pt") as f:
        for key in f.keys():
            if map_location == "meta":
                # Meta device: only create tensor metadata
                slice = f.get_slice(key)
                dtype = parse_dtype(slice.get_dtype())
                shape = slice.get_shape()
                state_dict[key] = create_empty_tensor(shape, dtype, device="meta")
            else:
                # Regular loading: read tensor and move to device
                tensor = f.get_tensor(key)
                state_dict[key] = tensor.to(map_location)

    return state_dict

function load_pytorch_checkpoint(checkpoint_file, map_location, weights_only):
    # Safety check for pickle deserialization
    if weights_only:
        validate_safe_to_load(checkpoint_file)

    # Use memory mapping if file is zip-based and not loading to meta
    use_mmap = (is_zipfile(checkpoint_file) and
                map_location != "meta")

    # Load with PyTorch
    state_dict = torch.load(
        checkpoint_file,
        map_location=map_location,
        weights_only=weights_only,
        mmap=use_mmap
    )

    return state_dict
```

'''Key Properties''':
* '''Format Agnostic''': Same interface regardless of serialization format
* '''Memory Efficiency''': Support for memory-mapped files and lazy loading
* '''Safety''': Safetensors prevents code execution vulnerabilities
* '''Device Control''': Explicit control over tensor placement

'''Memory Considerations''':
For a model with N parameters, each of size S bytes:
* '''Full Load''': Requires N × S bytes of memory
* '''Memory-Mapped''': Requires minimal memory, reads on-demand
* '''Meta Device''': Requires only metadata (< 1KB per tensor)

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_transformers_Weight_loading]]
