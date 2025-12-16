# Principle: Gradient Checkpointing

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|Training Deep Nets with Sublinear Memory Cost|https://arxiv.org/abs/1604.06174]]
* [[source::Doc|PyTorch Checkpoint Documentation|https://pytorch.org/docs/stable/checkpoint.html]]
* [[source::Blog|Memory-Efficient Training|https://huggingface.co/docs/transformers/perf_train_gpu_one]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Memory_Optimization]], [[domain::Training]]
|-
! Last Updated
| [[last_updated::2025-12-15 20:00 GMT]]
|}

== Overview ==
Memory optimization technique that trades compute for memory by selectively discarding intermediate activations during the forward pass and recomputing them during backpropagation.

=== Description ===
Gradient checkpointing addresses the memory bottleneck in training deep networks. Standard backpropagation stores all intermediate activations, consuming memory proportional to model depth. Checkpointing selectively saves only checkpoint activations, recomputing the rest as needed.

'''Memory Savings:'''
- Standard: O(L) memory for L layers
- Checkpointed: O(√L) memory with √L checkpoints
- Unsloth "unsloth" mode: Further optimized with selective layer checkpointing

'''Unsloth Optimization:'''
The `use_gradient_checkpointing="unsloth"` mode provides:
- Smart checkpoint placement based on layer memory profiles
- Reduced recomputation overhead vs standard checkpointing
- Compatibility with LoRA training (only active adapters participate)

=== Usage ===
Use gradient checkpointing when:
- Training large models on limited VRAM
- Batch size is memory-constrained
- Model depth exceeds memory capacity
- Fine-tuning with QLoRA on consumer GPUs

'''Configuration:'''
- `"unsloth"` - Recommended for Unsloth training
- `True` - Standard PyTorch checkpointing
- `False` - No checkpointing (maximum speed, most memory)

== Theoretical Basis ==
'''Standard Forward-Backward:'''

<syntaxhighlight lang="python">
# Without checkpointing: store all activations
def standard_forward_backward(x, layers):
    activations = [x]

    # Forward: O(L) memory for activations
    for layer in layers:
        x = layer(x)
        activations.append(x)  # Store for backward

    loss = compute_loss(x)

    # Backward: use stored activations
    for i, layer in enumerate(reversed(layers)):
        grad = layer.backward(grad, activations[-(i+2)])

    return grad
</syntaxhighlight>

'''Checkpointed Forward-Backward:'''
<syntaxhighlight lang="python">
# With checkpointing: O(√L) memory
def checkpointed_forward_backward(x, layers, checkpoint_every=4):
    checkpoints = {0: x}
    num_layers = len(layers)

    # Forward pass: only save checkpoint activations
    for i, layer in enumerate(layers):
        x = layer(x)
        if (i + 1) % checkpoint_every == 0:
            checkpoints[i + 1] = x

    loss = compute_loss(x)

    # Backward pass: recompute between checkpoints
    grad = loss.backward()
    for i in range(num_layers - 1, -1, -1):
        # Find nearest previous checkpoint
        ckpt_idx = (i // checkpoint_every) * checkpoint_every
        ckpt_activation = checkpoints[ckpt_idx]

        # Recompute forward from checkpoint to current layer
        recomputed = ckpt_activation
        for j in range(ckpt_idx, i):
            recomputed = layers[j](recomputed)

        # Now compute gradient
        grad = layers[i].backward(grad, recomputed)

    return grad
</syntaxhighlight>

'''Memory vs Compute Trade-off:'''
<math>
Memory = O(\frac{L}{k}) + O(k)
</math>

Where L is layers and k is checkpoint interval. Optimal when k = √L.

'''Unsloth Gradient Checkpointing:'''
<syntaxhighlight lang="python">
# Unsloth's optimized implementation
def unsloth_gradient_checkpointing(model):
    """Apply Unsloth's memory-efficient checkpointing."""

    # Only checkpoint decoder layers, not embeddings
    for layer in model.model.layers:
        # Use torch.utils.checkpoint with optimizations
        layer.forward = checkpoint_wrapper(
            layer.forward,
            use_reentrant=False,  # Avoid issues with LoRA
            preserve_rng_state=True,
        )

    # Don't checkpoint LoRA adapters (they're small)
    # Don't checkpoint attention (Unsloth fused kernels handle this)

    return model

# Benefits over standard checkpointing:
# 1. Selective: Only heavy decoder layers
# 2. Compatible: Works with fused Triton kernels
# 3. Efficient: Minimal recomputation of LoRA paths
</syntaxhighlight>

'''Practical Memory Impact:'''
| Model | Standard | Checkpointed | Savings |
|-------|----------|--------------|---------|
| 7B Llama | 28GB | 8GB | 3.5x |
| 13B Llama | 52GB | 12GB | 4.3x |
| 70B Llama | 280GB | 32GB | 8.7x |

(Estimates with batch_size=1, fp16)

== Related Pages ==
=== Implemented By ===
* [[implemented_by::Implementation:unslothai_unsloth_FastLanguageModel]]
* [[implemented_by::Implementation:unslothai_unsloth_FastVisionModel]]
* [[implemented_by::Implementation:unslothai_unsloth_UnslothTrainer]]

=== Tips and Tricks ===
