# Implementation: Gpt2

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PicoGPT|https://github.com/jaymody/picoGPT]]
* [[source::Paper|Language Models are Unsupervised Multitask Learners|https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::NLP]], [[domain::Model_Architecture]]
|-
! Last Updated
| [[last_updated::2026-01-15 19:00 GMT]]
|}

== Overview ==

Concrete tool for GPT-2 forward pass inference in pure NumPy provided by PicoGPT.

=== Description ===

The `gpt2()` function implements the complete forward pass of a GPT-2 model using only NumPy operations. It takes a sequence of token IDs and returns logits (unnormalized log-probabilities) over the vocabulary for each position.

The implementation follows the pre-norm Transformer architecture:
1. Sum token embeddings (wte) and positional embeddings (wpe)
2. Pass through N transformer blocks
3. Apply final layer normalization
4. Project to vocabulary via weight tying (multiply by transposed token embeddings)

This is an educational implementation focused on clarity over performance - no GPU acceleration, no KV caching, no optimizations.

=== Usage ===

Use `gpt2()` when:
- Performing a single forward pass for inference
- Understanding GPT-2 architecture by reading the code
- Building minimal transformer implementations for learning

This function is called inside the `generate()` loop for autoregressive text generation.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/jaymody/picoGPT picoGPT]
* '''File:''' gpt2.py
* '''Lines:''' 73-83

=== Signature ===
<syntaxhighlight lang="python">
def gpt2(
    inputs: list[int],
    wte: np.ndarray,
    wpe: np.ndarray,
    blocks: list[dict],
    ln_f: dict,
    n_head: int
) -> np.ndarray:
    """
    GPT-2 forward pass.

    Args:
        inputs: list[int] - Token IDs, length n_seq
        wte: np.ndarray - Token embeddings [n_vocab, n_embd]
        wpe: np.ndarray - Positional embeddings [n_ctx, n_embd]
        blocks: list[dict] - List of transformer block weights
        ln_f: dict - Final layer norm params {"g": ..., "b": ...}
        n_head: int - Number of attention heads

    Returns:
        np.ndarray - Logits of shape [n_seq, n_vocab]
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from gpt2 import gpt2
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| inputs || list[int] || Yes || Token IDs to process, length n_seq
|-
| wte || np.ndarray || Yes || Token embeddings, shape [n_vocab, n_embd]
|-
| wpe || np.ndarray || Yes || Positional embeddings, shape [n_ctx, n_embd]
|-
| blocks || list[dict] || Yes || List of n_layer transformer block weight dicts
|-
| ln_f || dict || Yes || Final layer norm: {"g": [n_embd], "b": [n_embd]}
|-
| n_head || int || Yes || Number of attention heads for multi-head attention
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| logits || np.ndarray || Output logits, shape [n_seq, n_vocab]. Not softmaxed!
|}

== Usage Examples ==

=== Basic Forward Pass ===
<syntaxhighlight lang="python">
import numpy as np
from utils import load_encoder_hparams_and_params
from gpt2 import gpt2

# Load model
encoder, hparams, params = load_encoder_hparams_and_params("124M", "models")

# Encode some text
tokens = encoder.encode("Hello, world")
print(f"Input tokens: {tokens}")  # [15496, 11, 995]

# Forward pass
logits = gpt2(
    inputs=tokens,
    wte=params["wte"],
    wpe=params["wpe"],
    blocks=params["blocks"],
    ln_f=params["ln_f"],
    n_head=hparams["n_head"]
)

print(f"Logits shape: {logits.shape}")  # (3, 50257) - [n_seq, n_vocab]
</syntaxhighlight>

=== Getting Next Token Prediction ===
<syntaxhighlight lang="python">
from gpt2 import gpt2
import numpy as np

# Forward pass
logits = gpt2(tokens, **params, n_head=hparams["n_head"])

# Get probabilities for last position
last_logits = logits[-1]  # [n_vocab]
probs = np.exp(last_logits) / np.sum(np.exp(last_logits))  # softmax

# Greedy: pick highest probability token
next_token = np.argmax(last_logits)
print(f"Next token ID: {next_token}")
print(f"Next token text: '{encoder.decode([next_token])}'")
print(f"Probability: {probs[next_token]:.4f}")

# Top 5 predictions
top5 = np.argsort(last_logits)[-5:][::-1]
for tok in top5:
    print(f"  {tok}: '{encoder.decode([tok])}' ({probs[tok]:.4f})")
</syntaxhighlight>

=== Understanding the Output Shape ===
<syntaxhighlight lang="python">
# The output has logits for EVERY position, not just the last
tokens = encoder.encode("The cat sat on")  # 4 tokens
logits = gpt2(tokens, **params, n_head=hparams["n_head"])

print(f"Input length: {len(tokens)}")    # 4
print(f"Output shape: {logits.shape}")   # (4, 50257)

# logits[0] = prediction for position 1 (given token 0)
# logits[1] = prediction for position 2 (given tokens 0-1)
# logits[2] = prediction for position 3 (given tokens 0-2)
# logits[3] = prediction for position 4 (given tokens 0-3) <- this is what we use for generation

# For autoregressive generation, we only need logits[-1]
# The other positions are useful for computing training loss
</syntaxhighlight>

== Internal Functions ==

The `gpt2()` function calls these helper functions:

=== transformer_block ===
<syntaxhighlight lang="python">
# gpt2.py:63-70
def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):
    """Single transformer block with pre-norm residual connections."""
    x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head)
    x = x + ffn(layer_norm(x, **ln_2), **mlp)
    return x
</syntaxhighlight>

=== mha (Multi-Head Attention) ===
<syntaxhighlight lang="python">
# gpt2.py:38-60
def mha(x, c_attn, c_proj, n_head):
    """Multi-head causal self-attention."""
    # Projects to Q,K,V, splits into heads, applies attention, merges heads
</syntaxhighlight>

=== ffn (Feed-Forward Network) ===
<syntaxhighlight lang="python">
# gpt2.py:24-31
def ffn(x, c_fc, c_proj):
    """Position-wise feed-forward network with GELU activation."""
    # Expands to 4x hidden dim, applies GELU, projects back
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Jaymody_PicoGPT_Transformer_Architecture]]

=== Requires Environment ===
* [[requires_env::Environment:Jaymody_PicoGPT_Python_Dependencies]]

=== Uses Heuristics ===
* [[uses_heuristic::Heuristic:Jaymody_PicoGPT_Causal_Masking_Large_Negative]]
* [[uses_heuristic::Heuristic:Jaymody_PicoGPT_Pre_Norm_Architecture]]
* [[uses_heuristic::Heuristic:Jaymody_PicoGPT_Weight_Tying_Embeddings]]
* [[uses_heuristic::Heuristic:Jaymody_PicoGPT_Stable_Softmax]]
