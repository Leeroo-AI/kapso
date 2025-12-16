# Principle: Low-Rank Adaptation (LoRA)

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|LoRA: Low-Rank Adaptation of Large Language Models|https://arxiv.org/abs/2106.09685]]
* [[source::Paper|QLoRA: Efficient Finetuning of Quantized LLMs|https://arxiv.org/abs/2305.14314]]
* [[source::Blog|Hugging Face PEFT Documentation|https://huggingface.co/docs/peft]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Parameter_Efficient_Fine_Tuning]], [[domain::LLMs]], [[domain::Transfer_Learning]]
|-
! Last Updated
| [[last_updated::2025-12-15 20:00 GMT]]
|}

== Overview ==
Parameter-efficient fine-tuning technique that freezes pre-trained model weights and injects trainable low-rank decomposition matrices into transformer layers, enabling adaptation with dramatically fewer trainable parameters.

=== Description ===
Low-Rank Adaptation (LoRA) addresses the challenge of fine-tuning large language models by making a key observation: the change in weights during fine-tuning has a low intrinsic rank. Instead of updating the full weight matrix W, LoRA decomposes the update into two smaller matrices:

'''Key Components:'''
1. **Frozen Base Weights (W₀)** - Original pre-trained weights remain unchanged
2. **Low-Rank Matrices (A, B)** - Two smaller matrices where BA approximates ΔW
3. **Scaling Factor (α/r)** - Controls the magnitude of the adaptation

'''Advantages over full fine-tuning:'''
- **Memory Efficient:** Only stores A and B matrices (r × d and d × r instead of d × d)
- **No Inference Latency:** Matrices can be merged into base weights post-training
- **Task Switching:** Multiple LoRA adapters can be swapped without reloading base model
- **Preserves Pre-training:** Frozen weights prevent catastrophic forgetting

'''Target Modules:'''
In transformers, LoRA is typically applied to:
- Attention projections: q_proj, k_proj, v_proj, o_proj
- MLP layers: gate_proj, up_proj, down_proj
- Optionally: embed_tokens, lm_head (for vocabulary adaptation)

=== Usage ===
Use LoRA when:
- Fine-tuning large models (7B+ parameters) with limited VRAM
- You need to maintain multiple task-specific adapters
- Training data is limited and you want to prevent overfitting
- Fast iteration on experiments is needed (smaller checkpoints)

'''Configuration Guidelines:'''
- **Rank (r):** Start with 8-16, increase to 64-128 for complex tasks
- **Alpha (α):** Typically set equal to r, or use α=2r for stronger adaptation
- **Target Modules:** Include attention + MLP for best results
- **Dropout:** 0 is optimized for Unsloth (uses fused kernels)

== Theoretical Basis ==
'''Low-Rank Decomposition:'''

For a pre-trained weight matrix W₀ ∈ ℝ^(d×k), LoRA represents the weight update as:

<math>
W = W_0 + \Delta W = W_0 + BA
</math>

Where:
- B ∈ ℝ^(d×r) (initialized to zeros)
- A ∈ ℝ^(r×k) (initialized from Gaussian)
- r << min(d, k) is the rank

'''Forward Pass:'''
<syntaxhighlight lang="python">
def forward(x, W0, A, B, alpha, r):
    """LoRA forward pass with scaling."""
    # Original computation + low-rank adaptation
    h = x @ W0.T + (x @ A.T @ B.T) * (alpha / r)
    return h

# Equivalently, for inference (merge weights):
def merge_lora(W0, A, B, alpha, r):
    """Merge LoRA weights for inference."""
    return W0 + (B @ A) * (alpha / r)
</syntaxhighlight>

'''Parameter Count:'''
<math>
Parameters_{LoRA} = 2 \times r \times d \times L
</math>

Where L is the number of target layers.

For a 7B model with r=16 on attention layers:
- Full fine-tuning: 7B parameters
- LoRA: ~50M parameters (~0.7% of full)

'''Why Low-Rank Works:'''
The weight updates during fine-tuning exhibit low intrinsic dimensionality. Even for tasks that seem complex, the necessary changes to weights can be captured by matrices of rank 8-64, far below the thousands of dimensions in actual weight matrices.

<syntaxhighlight lang="python">
# Rank selection heuristics:
def select_lora_rank(task_complexity, model_size):
    """Guidelines for LoRA rank selection."""
    if task_complexity == "simple":  # Sentiment, classification
        return 8
    elif task_complexity == "medium":  # QA, summarization
        return 16 if model_size < "7B" else 32
    elif task_complexity == "complex":  # Code generation, math
        return 64 if model_size < "7B" else 128
    return 16  # Safe default
</syntaxhighlight>

== Related Pages ==
=== Implemented By ===
* [[implemented_by::Implementation:unslothai_unsloth_FastLanguageModel]]
* [[implemented_by::Implementation:unslothai_unsloth_FastVisionModel]]

=== Tips and Tricks ===
