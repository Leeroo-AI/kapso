# Implementation: Generate

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PicoGPT|https://github.com/jaymody/picoGPT]]
* [[source::Blog|How to generate text with GPT-2|https://huggingface.co/blog/how-to-generate]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::NLP]], [[domain::Text_Generation]], [[domain::Sampling]]
|-
! Last Updated
| [[last_updated::2026-01-14 10:00 GMT]]
|}

== Overview ==

Concrete tool for autoregressive text generation using greedy decoding provided by the PicoGPT repository.

=== Description ===

The `generate` function implements the autoregressive generation loop for GPT-2:

1. **Iterates** for the specified number of tokens to generate
2. **Runs forward pass** through the GPT-2 model
3. **Selects next token** using greedy decoding (argmax)
4. **Appends** the selected token to the input sequence
5. **Displays progress** using tqdm progress bar

This implementation uses greedy decoding for simplicity—always selecting the highest probability token. While deterministic, this can lead to repetitive or generic text compared to sampling methods.

=== Usage ===

Use this function to generate text continuations from a tokenized prompt. Requires loaded model parameters and a list of input token IDs from encoding.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/jaymody/picoGPT PicoGPT]
* '''File:''' gpt2.py
* '''Lines:''' 86-94

=== Signature ===
<syntaxhighlight lang="python">
def generate(
    inputs: List[int],
    params: dict,
    n_head: int,
    n_tokens_to_generate: int
) -> List[int]:
    """
    Generate tokens autoregressively using greedy decoding.

    Args:
        inputs: Initial token IDs (prompt). Modified in-place!
        params: Model parameters dict (wte, wpe, blocks, ln_f).
        n_head: Number of attention heads.
        n_tokens_to_generate: Number of new tokens to generate.

    Returns:
        List of generated token IDs (not including prompt).

    Note:
        The inputs list is modified in-place during generation.
        Pass a copy if you need to preserve the original.
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from gpt2 import generate
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| inputs || List[int] || Yes || Initial token IDs (prompt). '''Warning: Modified in-place!'''
|-
| params || dict || Yes || Model parameters: {"wte", "wpe", "blocks", "ln_f"}
|-
| n_head || int || Yes || Number of attention heads (from hparams)
|-
| n_tokens_to_generate || int || Yes || How many new tokens to generate
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| (return) || List[int] || Generated token IDs only (length = n_tokens_to_generate)
|}

'''Side effect:''' The `inputs` list is mutated to include all generated tokens. After calling, `inputs` contains the full sequence (prompt + generated).

== Usage Examples ==

=== Basic Generation ===
<syntaxhighlight lang="python">
from gpt2 import generate
from utils import load_encoder_hparams_and_params

# Load model
encoder, hparams, params = load_encoder_hparams_and_params("124M", "models")

# Encode prompt
prompt = "The meaning of life is"
input_ids = encoder.encode(prompt)
print(f"Prompt: '{prompt}'")
print(f"Prompt tokens: {len(input_ids)}")

# Generate (note: input_ids is modified in-place)
output_ids = generate(
    inputs=input_ids,
    params=params,
    n_head=hparams["n_head"],
    n_tokens_to_generate=20
)

# Decode only the generated tokens
generated_text = encoder.decode(output_ids)
print(f"Generated: '{generated_text}'")

# Full output (prompt + generated)
full_text = encoder.decode(input_ids)
print(f"Full output: '{full_text}'")
</syntaxhighlight>

=== Preserving Original Input ===
<syntaxhighlight lang="python">
from gpt2 import generate
from utils import load_encoder_hparams_and_params

encoder, hparams, params = load_encoder_hparams_and_params("124M", "models")

prompt = "Once upon a time"
original_ids = encoder.encode(prompt)

# Pass a copy to preserve original
input_ids = original_ids.copy()
output_ids = generate(input_ids, params, hparams["n_head"], 30)

# original_ids is unchanged
assert len(original_ids) == len(encoder.encode(prompt))
</syntaxhighlight>

=== Using the main() Function ===
<syntaxhighlight lang="python">
from gpt2 import main

# The main function wraps everything together:
# 1. Loads model
# 2. Encodes prompt
# 3. Checks sequence length
# 4. Generates tokens
# 5. Decodes output

output = main(
    prompt="The quick brown fox",
    n_tokens_to_generate=40,
    model_size="124M",
    models_dir="models"
)
print(output)
</syntaxhighlight>

=== Command Line Usage ===
<syntaxhighlight lang="bash">
# Run from terminal using python-fire CLI
python gpt2.py --prompt "Hello, my name is" --n_tokens_to_generate 50

# With different model size
python gpt2.py --prompt "Once upon a time" --model_size "355M" --n_tokens_to_generate 100
</syntaxhighlight>

=== Understanding the Generation Loop ===
<syntaxhighlight lang="python">
# The generate function implements this loop:
#
# for _ in range(n_tokens_to_generate):
#     logits = gpt2(inputs, **params, n_head=n_head)  # Forward pass
#     next_id = np.argmax(logits[-1])                 # Greedy: pick highest prob
#     inputs.append(int(next_id))                     # Add to sequence
#
# return inputs[-n_tokens_to_generate:]               # Return generated only

# Key characteristics:
# - Greedy decoding (argmax) - deterministic output
# - No KV caching - full forward pass each iteration
# - No temperature/top-k/top-p sampling
# - Progress bar via tqdm
</syntaxhighlight>

=== Generation Speed Considerations ===
<syntaxhighlight lang="python">
# PicoGPT uses naive generation (no optimization):
#
# For sequence length L and generating G tokens:
# - Each step processes the full sequence: O(L²) per step
# - Total: O(G × (L+G)²) time complexity
#
# For GPT-2 124M with L=100, G=50:
# - Without KV cache: ~150 forward passes, each over growing sequence
# - With KV cache (not implemented): Each step only computes new token
#
# This is educational code - production systems use:
# - KV caching
# - Batched generation
# - GPU acceleration
# - Speculative decoding
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Jaymody_PicoGPT_Autoregressive_Generation]]

=== Requires Environment ===
* [[requires_env::Environment:Jaymody_PicoGPT_Python_Dependencies]]

=== Uses Heuristics ===
* [[uses_heuristic::Heuristic:Jaymody_PicoGPT_Greedy_Decoding_Tradeoffs]]
* [[uses_heuristic::Heuristic:Jaymody_PicoGPT_No_KV_Cache_Performance]]
* [[uses_heuristic::Heuristic:Jaymody_PicoGPT_Context_Length_Limits]]
