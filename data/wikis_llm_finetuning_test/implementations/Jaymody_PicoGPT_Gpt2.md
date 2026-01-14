# Implementation: Gpt2

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PicoGPT|https://github.com/jaymody/picoGPT]]
* [[source::Paper|Attention Is All You Need|https://arxiv.org/abs/1706.03762]]
* [[source::Blog|The Illustrated GPT-2|https://jalammar.github.io/illustrated-gpt2/]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::NLP]], [[domain::Transformers]], [[domain::Attention]]
|-
! Last Updated
| [[last_updated::2026-01-14 10:00 GMT]]
|}

== Overview ==

Concrete tool for computing the GPT-2 transformer forward pass in pure NumPy provided by the PicoGPT repository.

=== Description ===

The `gpt2` function implements the complete forward pass of the GPT-2 architecture:

1. **Embedding lookup:** Combines token and positional embeddings
2. **Transformer blocks:** Applies N layers of attention and feed-forward networks
3. **Output projection:** Projects final hidden states to vocabulary logits via weight tying

This implementation is notable for its educational clarity—all operations are explicit NumPy calls with shape annotations in comments. It demonstrates that transformer inference requires only basic linear algebra operations.

=== Usage ===

Use this function to compute logits for a sequence of tokens. Typically called by the `generate` function in an autoregressive loop, or directly for single-pass inference tasks like perplexity computation.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/jaymody/picoGPT PicoGPT]
* '''File:''' gpt2.py
* '''Lines:''' 73-83

=== Signature ===
<syntaxhighlight lang="python">
def gpt2(
    inputs: List[int],
    wte: np.ndarray,
    wpe: np.ndarray,
    blocks: List[dict],
    ln_f: dict,
    n_head: int
) -> np.ndarray:
    """
    GPT-2 forward pass.

    Args:
        inputs: List of token IDs, shape [n_seq].
        wte: Token embedding matrix, shape [n_vocab, n_embd].
        wpe: Position embedding matrix, shape [n_ctx, n_embd].
        blocks: List of transformer block parameters (n_layer blocks).
        ln_f: Final layer normalization parameters {"g": ..., "b": ...}.
        n_head: Number of attention heads.

    Returns:
        Logits array of shape [n_seq, n_vocab].
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
| inputs || List[int] || Yes || Token IDs, length n_seq (must be < n_ctx)
|-
| wte || np.ndarray || Yes || Token embeddings [n_vocab, n_embd]
|-
| wpe || np.ndarray || Yes || Position embeddings [n_ctx, n_embd]
|-
| blocks || List[dict] || Yes || Transformer block parameters (n_layer dicts)
|-
| ln_f || dict || Yes || Final layer norm: {"g": [n_embd], "b": [n_embd]}
|-
| n_head || int || Yes || Number of attention heads
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| (return) || np.ndarray || Logits of shape [n_seq, n_vocab] (unnormalized log-probabilities)
|}

**Block parameter structure:**
<syntaxhighlight lang="python">
# Each block in blocks list:
{
    "ln_1": {"g": [n_embd], "b": [n_embd]},      # Layer norm 1
    "attn": {
        "c_attn": {"w": [n_embd, 3*n_embd], "b": [3*n_embd]},  # QKV projection
        "c_proj": {"w": [n_embd, n_embd], "b": [n_embd]}        # Output projection
    },
    "ln_2": {"g": [n_embd], "b": [n_embd]},      # Layer norm 2
    "mlp": {
        "c_fc": {"w": [n_embd, 4*n_embd], "b": [4*n_embd]},     # Up projection
        "c_proj": {"w": [4*n_embd, n_embd], "b": [n_embd]}      # Down projection
    }
}
</syntaxhighlight>

== Usage Examples ==

=== Direct Forward Pass ===
<syntaxhighlight lang="python">
import numpy as np
from gpt2 import gpt2
from utils import load_encoder_hparams_and_params

# Load model
encoder, hparams, params = load_encoder_hparams_and_params("124M", "models")

# Tokenize input
text = "The capital of France is"
input_ids = encoder.encode(text)

# Run forward pass
logits = gpt2(
    inputs=input_ids,
    wte=params["wte"],
    wpe=params["wpe"],
    blocks=params["blocks"],
    ln_f=params["ln_f"],
    n_head=hparams["n_head"]
)

print(f"Input shape: {len(input_ids)}")           # e.g., 5
print(f"Output shape: {logits.shape}")            # (5, 50257)

# Get next token prediction
next_token_logits = logits[-1]  # Last position
next_token_id = np.argmax(next_token_logits)
next_token = encoder.decode([next_token_id])
print(f"Predicted next token: '{next_token}'")    # " Paris"
</syntaxhighlight>

=== Using Params Dict Unpacking ===
<syntaxhighlight lang="python">
from gpt2 import gpt2
from utils import load_encoder_hparams_and_params

encoder, hparams, params = load_encoder_hparams_and_params("124M", "models")

input_ids = encoder.encode("Hello world")

# Unpack params dict directly (matches function signature)
logits = gpt2(input_ids, **params, n_head=hparams["n_head"])
</syntaxhighlight>

=== Computing Probabilities ===
<syntaxhighlight lang="python">
import numpy as np
from gpt2 import gpt2, softmax
from utils import load_encoder_hparams_and_params

encoder, hparams, params = load_encoder_hparams_and_params("124M", "models")

text = "The quick brown"
input_ids = encoder.encode(text)

# Get logits
logits = gpt2(input_ids, **params, n_head=hparams["n_head"])

# Convert to probabilities
probs = softmax(logits[-1])

# Top-5 most likely next tokens
top5_ids = np.argsort(probs)[-5:][::-1]
for tid in top5_ids:
    token = encoder.decode([tid])
    print(f"'{token}': {probs[tid]:.4f}")
</syntaxhighlight>

=== Understanding Internal Shapes ===
<syntaxhighlight lang="python">
# Shape annotations from the code:
#
# gpt2(inputs, wte, wpe, blocks, ln_f, n_head):
#   x = wte[inputs] + wpe[range(len(inputs))]
#   # [n_seq] -> [n_seq, n_embd]
#
#   for block in blocks:
#       x = transformer_block(x, **block, n_head=n_head)
#       # [n_seq, n_embd] -> [n_seq, n_embd]
#
#   x = layer_norm(x, **ln_f)
#   # [n_seq, n_embd] -> [n_seq, n_embd]
#
#   return x @ wte.T
#   # [n_seq, n_embd] -> [n_seq, n_vocab]

# For GPT-2 124M:
# - n_embd = 768
# - n_vocab = 50257
# - n_layer = 12
# - n_head = 12
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Jaymody_PicoGPT_Transformer_Forward_Pass]]

=== Requires Environment ===
* [[requires_env::Environment:Jaymody_PicoGPT_Python_Dependencies]]

=== Uses Heuristics ===
* [[uses_heuristic::Heuristic:Jaymody_PicoGPT_No_KV_Cache_Performance]]
* [[uses_heuristic::Heuristic:Jaymody_PicoGPT_Context_Length_Limits]]
