{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|Transformers Model Loading|https://huggingface.co/docs/transformers/main_classes/model]]
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Accelerate Documentation|https://huggingface.co/docs/accelerate/]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Model_Management]], [[domain::Distributed_Computing]]
|-
! Last Updated
| [[last_updated::2025-12-18 00:00 GMT]]
|}

== Overview ==
Device Placement is the strategic assignment of model components to computational devices (CPUs, GPUs, disk) to optimize memory usage and inference performance.

=== Description ===
Modern large language models often exceed the memory capacity of a single GPU. Device Placement solves this by intelligently distributing model layers and parameters across multiple devices, including CPU RAM and even disk storage for offloading. This principle handles the complexity of determining optimal device assignments, managing cross-device communication during forward passes, implementing memory offloading strategies, and ensuring that the model remains functionally correct despite being split across heterogeneous hardware.

The principle encompasses automatic device map generation (where the system analyzes model size and available memory to create optimal placements), manual device map specification (for fine-grained control), and the mechanics of dispatching tensors and modules to their assigned devices while managing the data transfers needed during computation.

=== Usage ===
Device Placement should be applied after model instantiation and weight loading, as the final step in model preparation. Use it when working with models that don't fit in single-device memory, when optimizing multi-GPU inference, or when implementing memory-efficient inference on resource-constrained hardware.

== Theoretical Basis ==

Device placement implements a constraint optimization problem:

1. '''Problem Definition''':
   * Given: Model M with layers L₁, L₂, ..., Lₙ
   * Given: Devices D₁ (GPU1), D₂ (GPU2), ..., Dₖ (CPU), Dₖ₊₁ (Disk)
   * Given: Memory capacities C₁, C₂, ..., Cₖ₊₁
   * Goal: Assign each layer Lᵢ to device Dⱼ such that:
     - Memory constraints satisfied: Σ(size(Lᵢ) for Lᵢ on Dⱼ) ≤ Cⱼ
     - Cross-device transfers minimized
     - Computation time minimized

2. '''Device Map Generation''':
```
function generate_device_map(model, available_devices):
    # Calculate memory requirements per layer
    layer_sizes = {}
    for layer in model.layers:
        layer_sizes[layer.name] = calculate_memory(layer)

    # Sort devices by compute capability (GPU > CPU > Disk)
    devices = sort_by_capability(available_devices)

    # Greedy assignment
    device_map = {}
    current_device = 0

    for layer_name, size in layer_sizes.items():
        # Check if current device has capacity
        while current_device < len(devices):
            if get_available_memory(devices[current_device]) >= size:
                device_map[layer_name] = devices[current_device]
                allocate_memory(devices[current_device], size)
                break
            else:
                current_device += 1

        if layer_name not in device_map:
            # Offload to disk if no device has space
            device_map[layer_name] = "disk"

    return device_map
```

3. '''Dispatch Mechanics''':
```
function dispatch_model(model, device_map, offload_dir):
    # Create execution hooks for cross-device transfers
    for module_name, device in device_map.items():
        module = get_module(model, module_name)

        if device == "disk":
            # Setup offload hooks
            create_offload_hook(module, offload_dir,
                              on_forward_start=load_to_device,
                              on_forward_end=offload_to_disk)
        else:
            # Move module to device
            module.to(device)

        # Create pre-forward hook for input transfer
        def pre_forward_hook(input):
            return transfer_to_device(input, device)

        module.register_forward_pre_hook(pre_forward_hook)

        # Create post-forward hook for output transfer
        def post_forward_hook(output):
            next_device = get_next_layer_device(module_name)
            return transfer_to_device(output, next_device)

        module.register_forward_hook(post_forward_hook)

    return model
```

4. '''Forward Pass with Device Placement''':
```
For a model with layers on different devices:
Input (GPU:0) → Layer1 (GPU:0) → [transfer] → Layer2 (GPU:1) → [transfer] → Layer3 (CPU) → Output (CPU)

Each transfer introduces latency: t_transfer = data_size / bandwidth
Total time = Σ(t_compute) + Σ(t_transfer)
```

'''Optimization Strategies''':
* '''Sequential Placement''': Minimize transfers by placing sequential layers together
* '''Layer Balancing''': Distribute layers evenly across GPUs
* '''Offload Priorities''': Place frequently-used layers on faster devices

'''Memory Trade-offs''':
* '''GPU-only''': Fast but limited by GPU memory
* '''GPU + CPU''': More capacity, slower due to PCIe transfers
* '''GPU + CPU + Disk''': Maximum capacity, slowest due to I/O

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_transformers_Accelerate_dispatch]]
