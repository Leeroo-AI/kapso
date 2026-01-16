# Implementation: Load_Gpt2_Params_From_Tf_Ckpt

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PicoGPT|https://github.com/jaymody/picoGPT]]
* [[source::Doc|TensorFlow Checkpoint API|https://www.tensorflow.org/api_docs/python/tf/train/load_variable]]
|-
! Domains
| [[domain::Model_Management]], [[domain::Data_Engineering]]
|-
! Last Updated
| [[last_updated::2026-01-15 19:00 GMT]]
|}

== Overview ==

Concrete tool for converting TensorFlow GPT-2 checkpoints to NumPy arrays provided by PicoGPT.

=== Description ===

The `load_gpt2_params_from_tf_ckpt` function reads a TensorFlow checkpoint file and converts all variables into a nested Python dictionary containing NumPy arrays. It parses the hierarchical variable names (e.g., `model/h0/attn/c_proj/w`) and restructures them into the format expected by the GPT-2 forward pass function.

The function handles the mapping from TensorFlow's flat variable namespace to the nested structure required for inference:
- `model/wte` → `params["wte"]`
- `model/wpe` → `params["wpe"]`
- `model/h{N}/...` → `params["blocks"][N][...]`
- `model/ln_f/...` → `params["ln_f"][...]`

=== Usage ===

Use this function when:
- Loading official OpenAI GPT-2 checkpoints for NumPy inference
- Converting TensorFlow checkpoints to a framework-agnostic format
- Debugging or inspecting model weights

This is typically called by `load_encoder_hparams_and_params` as part of the model loading pipeline.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/jaymody/picoGPT picoGPT]
* '''File:''' utils.py
* '''Lines:''' 44-65

=== Signature ===
<syntaxhighlight lang="python">
def load_gpt2_params_from_tf_ckpt(tf_ckpt_path: str, hparams: dict) -> dict:
    """
    Load GPT-2 parameters from a TensorFlow checkpoint into NumPy arrays.

    Args:
        tf_ckpt_path: str - Path to TensorFlow checkpoint (without file extension)
        hparams: dict - Hyperparameters dict containing 'n_layer' key

    Returns:
        dict - Nested dictionary with structure:
            {
                "wte": np.ndarray,      # Token embeddings [n_vocab, n_embd]
                "wpe": np.ndarray,      # Position embeddings [n_ctx, n_embd]
                "blocks": [             # List of transformer block weights
                    {
                        "ln_1": {"g": ..., "b": ...},
                        "attn": {"c_attn": {...}, "c_proj": {...}},
                        "ln_2": {"g": ..., "b": ...},
                        "mlp": {"c_fc": {...}, "c_proj": {...}}
                    },
                    ...
                ],
                "ln_f": {"g": ..., "b": ...}  # Final layer norm
            }
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from utils import load_gpt2_params_from_tf_ckpt
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| tf_ckpt_path || str || Yes || Path to TensorFlow checkpoint (prefix, without .data/.index extension)
|-
| hparams || dict || Yes || Hyperparameters dict, must contain "n_layer" key to know how many blocks to initialize
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| params || dict || Nested dict containing all model parameters as NumPy arrays
|-
| params["wte"] || np.ndarray || Token embeddings, shape [n_vocab, n_embd]
|-
| params["wpe"] || np.ndarray || Positional embeddings, shape [n_ctx, n_embd]
|-
| params["blocks"] || list[dict] || List of n_layer transformer block weight dicts
|-
| params["ln_f"] || dict || Final layer normalization weights {"g": ..., "b": ...}
|}

== Usage Examples ==

=== Basic Checkpoint Loading ===
<syntaxhighlight lang="python">
import json
import tensorflow as tf
from utils import load_gpt2_params_from_tf_ckpt

# Load hyperparameters
with open("models/124M/hparams.json") as f:
    hparams = json.load(f)
    # hparams = {"n_vocab": 50257, "n_ctx": 1024, "n_embd": 768,
    #            "n_head": 12, "n_layer": 12}

# Get checkpoint path
tf_ckpt_path = tf.train.latest_checkpoint("models/124M")
# tf_ckpt_path = "models/124M/model.ckpt"

# Load parameters
params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, hparams)

# Inspect structure
print(params.keys())  # dict_keys(['wte', 'wpe', 'blocks', 'ln_f'])
print(params["wte"].shape)  # (50257, 768) for 124M model
print(len(params["blocks"]))  # 12 for 124M model
</syntaxhighlight>

=== Inspecting Block Structure ===
<syntaxhighlight lang="python">
# Each block contains attention and MLP weights
block = params["blocks"][0]
print(block.keys())  # dict_keys(['ln_1', 'attn', 'ln_2', 'mlp'])

# Attention weights
print(block["attn"]["c_attn"]["w"].shape)  # (768, 2304) = n_embd -> 3*n_embd (Q,K,V)
print(block["attn"]["c_proj"]["w"].shape)  # (768, 768) = n_embd -> n_embd

# MLP weights
print(block["mlp"]["c_fc"]["w"].shape)    # (768, 3072) = n_embd -> 4*n_embd
print(block["mlp"]["c_proj"]["w"].shape)  # (3072, 768) = 4*n_embd -> n_embd
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Jaymody_PicoGPT_Weight_Conversion]]

=== Requires Environment ===
* [[requires_env::Environment:Jaymody_PicoGPT_Python_Dependencies]]
