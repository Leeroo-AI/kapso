# Implementation: Download_Gpt2_Files

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PicoGPT|https://github.com/jaymody/picoGPT]]
* [[source::Doc|OpenAI GPT-2 Model Card|https://github.com/openai/gpt-2/blob/master/model_card.md]]
|-
! Domains
| [[domain::Infrastructure]], [[domain::Model_Management]]
|-
! Last Updated
| [[last_updated::2026-01-15 19:00 GMT]]
|}

== Overview ==

Concrete tool for downloading GPT-2 checkpoint files from OpenAI's Azure blob storage provided by PicoGPT.

=== Description ===

The `download_gpt2_files` function retrieves all necessary files for a GPT-2 model variant from OpenAI's public Azure blob storage. It downloads seven files in total: the TensorFlow checkpoint (3 files), vocabulary files (encoder.json, vocab.bpe), hyperparameters (hparams.json), and checkpoint metadata.

The function uses streaming downloads with progress bars to handle the potentially large checkpoint files (124M variant is ~500MB, 1558M is ~6GB).

=== Usage ===

Use this function when:
- First-time setup of a GPT-2 model on a new machine
- The models directory is empty or missing required files
- You need to download a different model size variant

Note: This is typically called automatically by `load_encoder_hparams_and_params` when checkpoint files are not found locally.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/jaymody/picoGPT picoGPT]
* '''File:''' utils.py
* '''Lines:''' 13-41

=== Signature ===
<syntaxhighlight lang="python">
def download_gpt2_files(model_size: str, model_dir: str) -> None:
    """
    Download GPT-2 model files from OpenAI's Azure blob storage.

    Args:
        model_size: str - One of "124M", "355M", "774M", "1558M"
        model_dir: str - Local directory path to save the downloaded files

    Returns:
        None - Files are written to disk

    Raises:
        AssertionError: If model_size is not a valid GPT-2 variant
        requests.HTTPError: If download fails
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from utils import download_gpt2_files
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model_size || str || Yes || GPT-2 model variant: "124M", "355M", "774M", or "1558M"
|-
| model_dir || str || Yes || Local directory path where files will be saved
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| (none) || None || Function writes files to disk, returns nothing
|-
| checkpoint || Files || model.ckpt.data-00000-of-00001, model.ckpt.index, model.ckpt.meta
|-
| vocab || Files || encoder.json (token mappings), vocab.bpe (BPE merges)
|-
| config || Files || hparams.json (model hyperparameters), checkpoint (TF metadata)
|}

== Usage Examples ==

=== Basic Download ===
<syntaxhighlight lang="python">
from utils import download_gpt2_files
import os

# Create models directory if needed
model_dir = "models/124M"
os.makedirs(model_dir, exist_ok=True)

# Download the smallest GPT-2 model
download_gpt2_files("124M", model_dir)

# Files now available in models/124M/:
# - checkpoint
# - encoder.json
# - hparams.json
# - model.ckpt.data-00000-of-00001
# - model.ckpt.index
# - model.ckpt.meta
# - vocab.bpe
</syntaxhighlight>

=== Downloading Different Model Sizes ===
<syntaxhighlight lang="python">
from utils import download_gpt2_files
import os

# Available sizes and approximate download sizes:
# - "124M": ~500MB (smallest, fastest inference)
# - "355M": ~1.5GB
# - "774M": ~3GB
# - "1558M": ~6GB (largest, highest quality)

for size in ["124M", "355M"]:
    model_dir = f"models/{size}"
    os.makedirs(model_dir, exist_ok=True)
    print(f"Downloading GPT-2 {size}...")
    download_gpt2_files(size, model_dir)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Jaymody_PicoGPT_Model_Download]]

=== Requires Environment ===
* [[requires_env::Environment:Jaymody_PicoGPT_Python_Dependencies]]

=== Uses Heuristics ===
* [[uses_heuristic::Heuristic:Jaymody_PicoGPT_Streaming_Download_Large_Files]]
