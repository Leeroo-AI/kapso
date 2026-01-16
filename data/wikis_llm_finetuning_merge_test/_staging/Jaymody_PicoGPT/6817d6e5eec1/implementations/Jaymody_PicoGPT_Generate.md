# Implementation: Generate

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PicoGPT|https://github.com/jaymody/picoGPT]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::NLP]], [[domain::Text_Generation]]
|-
! Last Updated
| [[last_updated::2026-01-15 19:00 GMT]]
|}

== Overview ==

Concrete tool for autoregressive text generation using GPT-2 provided by PicoGPT.

=== Description ===

The `generate()` function implements the autoregressive generation loop for GPT-2. It takes an initial sequence of token IDs and iteratively generates new tokens by:

1. Running the full forward pass through the transformer (`gpt2()` function)
2. Taking the logits at the last position
3. Selecting the token with highest probability (greedy decoding)
4. Appending to the input sequence and repeating

The function uses a progress bar via `tqdm` to show generation progress.

=== Usage ===

Use `generate()` when:
- Performing text completion with a loaded GPT-2 model
- Building the inference pipeline after encoding input text
- Generating a fixed number of new tokens

Note: This implementation uses greedy decoding only. For sampling-based generation (temperature, top-k, top-p), you would need to modify this function.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/jaymody/picoGPT picoGPT]
* '''File:''' gpt2.py
* '''Lines:''' 86-94

=== Signature ===
<syntaxhighlight lang="python">
def generate(
    inputs: list[int],
    params: dict,
    n_head: int,
    n_tokens_to_generate: int
) -> list[int]:
    """
    Generate tokens autoregressively using GPT-2.

    Args:
        inputs: list[int] - Initial token IDs (prompt). Modified in-place!
        params: dict - Model parameters (wte, wpe, blocks, ln_f)
        n_head: int - Number of attention heads
        n_tokens_to_generate: int - Number of new tokens to generate

    Returns:
        list[int] - Only the newly generated token IDs (not the prompt)

    Note:
        - Uses greedy decoding (argmax)
        - The inputs list is modified in-place (prompt + generated tokens)
        - Shows progress bar during generation
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
| inputs || list[int] || Yes || Initial token IDs (prompt). '''Warning: modified in-place!'''
|-
| params || dict || Yes || Model weights dict with keys: wte, wpe, blocks, ln_f
|-
| n_head || int || Yes || Number of attention heads (12 for 124M, 16 for 355M, etc.)
|-
| n_tokens_to_generate || int || Yes || How many new tokens to generate
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| generated_ids || list[int] || List of newly generated token IDs (length = n_tokens_to_generate)
|}

== Usage Examples ==

=== Basic Generation ===
<syntaxhighlight lang="python">
from utils import load_encoder_hparams_and_params
from gpt2 import generate

# Load model
encoder, hparams, params = load_encoder_hparams_and_params("124M", "models")

# Encode prompt
prompt = "The quick brown fox"
input_ids = encoder.encode(prompt)

# Generate 20 new tokens
output_ids = generate(
    inputs=input_ids,
    params=params,
    n_head=hparams["n_head"],
    n_tokens_to_generate=20
)

# Decode output
output_text = encoder.decode(output_ids)
print(f"Generated: {output_text}")
</syntaxhighlight>

=== Full Pipeline Example ===
<syntaxhighlight lang="python">
from utils import load_encoder_hparams_and_params
from gpt2 import generate

# Load everything
encoder, hparams, params = load_encoder_hparams_and_params("124M", "models")

# User prompt
prompt = "In a shocking finding, scientists discovered a"

# Encode (make a copy since generate modifies in-place)
input_ids = encoder.encode(prompt)
print(f"Prompt: '{prompt}'")
print(f"Prompt tokens: {len(input_ids)}")

# Check context length
n_tokens_to_generate = 40
assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"], \
    f"Total length {len(input_ids) + n_tokens_to_generate} exceeds context {hparams['n_ctx']}"

# Generate
output_ids = generate(
    inputs=input_ids,
    params=params,
    n_head=hparams["n_head"],
    n_tokens_to_generate=n_tokens_to_generate
)

# Decode and print
output_text = encoder.decode(output_ids)
print(f"\n{prompt}{output_text}")
</syntaxhighlight>

=== Understanding In-Place Modification ===
<syntaxhighlight lang="python">
from gpt2 import generate

# Important: inputs is modified in-place!
input_ids = [15496, 11]  # "Hello,"
original_length = len(input_ids)

output_ids = generate(input_ids, params, n_head=12, n_tokens_to_generate=5)

print(f"Original input length: {original_length}")     # 2
print(f"Input list after generate: {len(input_ids)}")  # 7 (modified!)
print(f"Output list length: {len(output_ids)}")        # 5 (just the new tokens)

# If you need to preserve the original:
input_ids_copy = encoder.encode(prompt).copy()
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Jaymody_PicoGPT_Autoregressive_Generation]]

=== Requires Environment ===
* [[requires_env::Environment:Jaymody_PicoGPT_Python_Dependencies]]

=== Uses Heuristics ===
* [[uses_heuristic::Heuristic:Jaymody_PicoGPT_Sequence_Length_Validation]]
