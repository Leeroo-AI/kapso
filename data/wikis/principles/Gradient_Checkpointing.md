{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|Training Deep Nets with Sublinear Memory|https://arxiv.org/abs/1604.06174]]
* [[source::Doc|PyTorch Checkpointing|https://pytorch.org/docs/stable/checkpoint.html]]
* [[source::Repo|Unsloth GitHub|https://github.com/unslothai/unsloth]]
|-
! Domains
| [[domain::Optimization]], [[domain::Memory_Management]], [[domain::Deep_Learning]]
|-
! Last Updated
| [[last_updated::2025-12-12 00:00 GMT]]
|}

== Overview ==
Memory optimization technique that trades compute for memory by recomputing activations during backward pass instead of storing them.

=== Description ===
During standard backpropagation, all intermediate activations from the forward pass are stored for gradient computation. For deep networks, this consumes massive memory. Gradient checkpointing stores only selected activations (checkpoints) and recomputes the others during the backward pass. This reduces memory from O(n) to O(√n) for n layers, at the cost of one additional forward pass.

=== Usage ===
Use this principle when training deep models that don't fit in GPU memory. Essential for fine-tuning large language models (7B+ parameters) on consumer hardware. Apply when you see CUDA Out of Memory errors during training. Unsloth's "unsloth" mode provides enhanced gradient checkpointing with 30% additional savings.

== Theoretical Basis ==
'''Standard Training Memory:'''
For a model with L layers, each producing activations of size A:
\[
\text{Memory} = L \times A \quad \text{(all activations stored)}
\]

'''With Checkpointing:'''
Store only √L checkpoints, recompute intermediate activations:
\[
\text{Memory} \approx \sqrt{L} \times A
\]

'''Algorithm:'''
<syntaxhighlight lang="python">
# Conceptual gradient checkpointing
def checkpointed_forward(model, x, checkpoint_layers):
    activations = {}
    
    for i, layer in enumerate(model.layers):
        x = layer(x)
        if i in checkpoint_layers:
            activations[i] = x.detach()  # Store checkpoint
    
    return x, activations

def checkpointed_backward(model, grad, activations, checkpoint_layers):
    for i in reversed(range(len(model.layers))):
        if i in checkpoint_layers:
            # Recompute activations from checkpoint
            x = activations[nearest_checkpoint(i)]
            for j in range(nearest_checkpoint(i), i):
                x = model.layers[j](x)
        
        grad = backward_through_layer(model.layers[i], x, grad)
    
    return grad
</syntaxhighlight>

'''Trade-offs:'''
* Memory reduction: ~50-70% for standard, ~60-80% with Unsloth
* Compute overhead: ~20-33% (one extra forward pass per segment)
* Unsloth optimizes this with smart checkpointing and kernel fusion

== Related Pages ==
=== Implemented By ===
* [[implemented_by::Implementation:Unsloth_get_peft_model]]

=== Tips and Tricks ===
* [[uses_heuristic::Heuristic:Unsloth_Gradient_Checkpointing_Optimization]]
* [[uses_heuristic::Heuristic:Batch_Size_Optimization]]

