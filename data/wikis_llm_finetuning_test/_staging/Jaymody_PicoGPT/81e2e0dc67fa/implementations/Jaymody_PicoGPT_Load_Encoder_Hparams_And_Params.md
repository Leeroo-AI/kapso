# Implementation: Load_Encoder_Hparams_And_Params

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PicoGPT|https://github.com/jaymody/picoGPT]]
* [[source::Doc|TensorFlow Checkpoint API|https://www.tensorflow.org/api_docs/python/tf/train]]
|-
! Domains
| [[domain::Model_Management]], [[domain::Deep_Learning]], [[domain::NLP]]
|-
! Last Updated
| [[last_updated::2026-01-14 10:00 GMT]]
|}

== Overview ==

Concrete tool for loading GPT-2 model weights, hyperparameters, and tokenizer from TensorFlow checkpoints provided by the PicoGPT repository.

=== Description ===

The `load_encoder_hparams_and_params` function is the main entry point for model initialization in PicoGPT. It orchestrates:

1. **Checkpoint discovery:** Finds the TensorFlow checkpoint in the model directory
2. **Lazy downloading:** Downloads model files from OpenAI's Azure storage if not cached
3. **Encoder creation:** Loads BPE vocabulary and merge rules into an Encoder instance
4. **Hyperparameter loading:** Reads model configuration from `hparams.json`
5. **Weight parsing:** Converts TensorFlow checkpoint variables into nested NumPy arrays

The function returns all components needed to run inference: tokenizer, model config, and weights.

=== Usage ===

Import this function when initializing a GPT-2 model for text generation. It handles all model loading concerns including downloading, caching, and format conversion.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/jaymody/picoGPT PicoGPT]
* '''File:''' utils.py
* '''Lines:''' 68-82

=== Signature ===
<syntaxhighlight lang="python">
def load_encoder_hparams_and_params(model_size: str, models_dir: str) -> Tuple[Encoder, dict, dict]:
    """
    Load GPT-2 encoder, hyperparameters, and model parameters.

    Args:
        model_size: GPT-2 model variant. One of "124M", "355M", "774M", "1558M".
        models_dir: Directory to store/load model files.

    Returns:
        Tuple of:
            - encoder: Encoder instance for tokenization
            - hparams: dict with model hyperparameters (n_vocab, n_ctx, n_embd, n_head, n_layer)
            - params: dict with model weights (wte, wpe, blocks, ln_f)
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from utils import load_encoder_hparams_and_params
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model_size || str || Yes || GPT-2 variant: "124M", "355M", "774M", or "1558M"
|-
| models_dir || str || Yes || Directory path for model file storage
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| encoder || Encoder || BPE tokenizer with encode() and decode() methods
|-
| hparams || dict || Model hyperparameters: n_vocab, n_ctx, n_embd, n_head, n_layer
|-
| params || dict || Nested dict of NumPy arrays: wte, wpe, blocks, ln_f
|}

**Output params structure:**
<syntaxhighlight lang="python">
{
    "wte": np.ndarray,        # [n_vocab, n_embd] token embeddings
    "wpe": np.ndarray,        # [n_ctx, n_embd] position embeddings
    "blocks": [               # List of n_layer transformer blocks
        {
            "ln_1": {"g": np.ndarray, "b": np.ndarray},
            "attn": {
                "c_attn": {"w": np.ndarray, "b": np.ndarray},
                "c_proj": {"w": np.ndarray, "b": np.ndarray}
            },
            "ln_2": {"g": np.ndarray, "b": np.ndarray},
            "mlp": {
                "c_fc": {"w": np.ndarray, "b": np.ndarray},
                "c_proj": {"w": np.ndarray, "b": np.ndarray}
            }
        },
        # ... n_layer blocks
    ],
    "ln_f": {"g": np.ndarray, "b": np.ndarray}  # Final layer norm
}
</syntaxhighlight>

== Usage Examples ==

=== Basic Model Loading ===
<syntaxhighlight lang="python">
from utils import load_encoder_hparams_and_params

# Load the smallest GPT-2 model (124M parameters)
encoder, hparams, params = load_encoder_hparams_and_params("124M", "models")

# Inspect hyperparameters
print(f"Vocabulary size: {hparams['n_vocab']}")      # 50257
print(f"Context length: {hparams['n_ctx']}")         # 1024
print(f"Embedding dimension: {hparams['n_embd']}")   # 768
print(f"Number of heads: {hparams['n_head']}")       # 12
print(f"Number of layers: {hparams['n_layer']}")     # 12

# Check weight shapes
print(f"Token embeddings: {params['wte'].shape}")    # (50257, 768)
print(f"Position embeddings: {params['wpe'].shape}") # (1024, 768)
print(f"Number of blocks: {len(params['blocks'])}")  # 12
</syntaxhighlight>

=== Loading Different Model Sizes ===
<syntaxhighlight lang="python">
from utils import load_encoder_hparams_and_params

# Model sizes and their approximate VRAM requirements
model_configs = {
    "124M": "~500MB",
    "355M": "~1.5GB",
    "774M": "~3GB",
    "1558M": "~6GB"
}

# Load a specific model size
model_size = "355M"
encoder, hparams, params = load_encoder_hparams_and_params(model_size, "models")

# First call downloads files; subsequent calls load from cache
# Files are stored in: models/{model_size}/
</syntaxhighlight>

=== Full Inference Pipeline ===
<syntaxhighlight lang="python">
from utils import load_encoder_hparams_and_params
from gpt2 import generate

# 1. Load model
encoder, hparams, params = load_encoder_hparams_and_params("124M", "models")

# 2. Encode prompt
prompt = "The meaning of life is"
input_ids = encoder.encode(prompt)

# 3. Generate tokens
output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate=20)

# 4. Decode output
output_text = encoder.decode(output_ids)
print(output_text)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Jaymody_PicoGPT_Model_Loading]]

=== Requires Environment ===
* [[requires_env::Environment:Jaymody_PicoGPT_Python_Dependencies]]

=== Uses Heuristics ===
* [[uses_heuristic::Heuristic:Jaymody_PicoGPT_Model_Size_Memory_Requirements]]
