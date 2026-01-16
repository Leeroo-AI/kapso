# Environment: Python_Dependencies

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|picoGPT|https://github.com/jaymody/picoGPT]]
* [[source::Doc|requirements.txt|https://github.com/jaymody/picoGPT/blob/main/requirements.txt]]
|-
! Domains
| [[domain::NLP]], [[domain::Deep_Learning]], [[domain::Infrastructure]]
|-
! Last Updated
| [[last_updated::2026-01-15 19:00 GMT]]
|}

== Overview ==
Python 3.9 environment with NumPy, TensorFlow, and supporting packages for running GPT-2 inference.

=== Description ===
This environment provides the minimal dependencies required to run the PicoGPT implementation. It uses TensorFlow solely for loading the original GPT-2 checkpoint weights (not for training or inference), NumPy for all actual model computations, and the `regex` library for BPE tokenization. The environment supports both standard systems and Apple Silicon Macs via platform-specific TensorFlow packages.

=== Usage ===
Use this environment for any **Model Loading**, **Text Generation**, or **Tokenization** workflow using PicoGPT. It is the mandatory prerequisite for running all implementations in this repository: `Download_Gpt2_Files`, `Load_Gpt2_Params_From_Tf_Ckpt`, `Encoder`, `Encoder_Encode`, `Encoder_Decode`, `Gpt2`, and `Generate`.

== System Requirements ==
{| class="wikitable"
! Category !! Requirement !! Notes
|-
| OS || Linux, macOS, Windows || M1/M2 Macs require tensorflow-macos
|-
| Python || 3.9+ || Tested on Python 3.9.10
|-
| Disk || 5GB+ || For downloaded model checkpoints (124M-1558M)
|-
| RAM || 4GB+ || 1558M model requires more RAM
|}

== Dependencies ==
=== System Packages ===
* No system packages required beyond Python

=== Python Packages ===
* `numpy` == 1.24.1 (model code and weights)
* `regex` == 2017.4.5 (BPE tokenizer)
* `requests` == 2.27.1 (download GPT-2 files from OpenAI)
* `tqdm` == 4.64.0 (progress bar)
* `fire` == 0.5.0 (CLI creation)
* `tensorflow` == 2.11.0 (load GPT-2 weights from TF checkpoint; non-ARM Macs/Linux/Windows)
* `tensorflow-macos` == 2.11.0 (load GPT-2 weights on Apple Silicon)

== Credentials ==
The following environment variables may be needed:
* None required - all model files are downloaded from public Azure blob storage

== Quick Install ==
<syntaxhighlight lang="bash">
# Standard installation
pip install numpy==1.24.1 regex==2017.4.5 requests==2.27.1 tqdm==4.64.0 fire==0.5.0 tensorflow==2.11.0

# For Apple Silicon (M1/M2) Macs
pip install numpy==1.24.1 regex==2017.4.5 requests==2.27.1 tqdm==4.64.0 fire==0.5.0 tensorflow-macos==2.11.0

# Or simply
pip install -r requirements.txt
</syntaxhighlight>

== Code Evidence ==

Import statements from `utils.py:L1-10`:
<syntaxhighlight lang="python">
import json
import os
import re

import numpy as np
import requests
import tensorflow as tf
from tqdm import tqdm

from encoder import get_encoder
</syntaxhighlight>

Model size validation from `utils.py:L14`:
<syntaxhighlight lang="python">
assert model_size in ["124M", "355M", "774M", "1558M"]
</syntaxhighlight>

TensorFlow checkpoint loading from `utils.py:L54-55`:
<syntaxhighlight lang="python">
for name, _ in tf.train.list_variables(tf_ckpt_path):
    array = np.squeeze(tf.train.load_variable(tf_ckpt_path, name))
</syntaxhighlight>

Regex import from `encoder.py:L9`:
<syntaxhighlight lang="python">
import regex as re
</syntaxhighlight>

Platform-specific TensorFlow from `requirements.txt:L9-10`:
<syntaxhighlight lang="bash">
tensorflow==2.11.0; sys_platform != 'darwin' or platform_machine != 'arm64'
tensorflow-macos==2.11.0; sys_platform == 'darwin' and platform_machine == 'arm64'
</syntaxhighlight>

== Common Errors ==

{| class="wikitable"
|-
! Error Message !! Cause !! Solution
|-
|| `ModuleNotFoundError: No module named 'tensorflow'` || TensorFlow not installed || `pip install tensorflow==2.11.0`
|-
|| `ModuleNotFoundError: No module named 'regex'` || regex package not installed || `pip install regex==2017.4.5`
|-
|| `AssertionError` in model_size check || Invalid model size string || Use one of: "124M", "355M", "774M", "1558M"
|-
|| `requests.exceptions.ConnectionError` || Network issue downloading models || Check internet connection; files download from Azure blob storage
|-
|| TensorFlow crash on Apple Silicon || Using standard TensorFlow on M1/M2 || Use `pip install tensorflow-macos==2.11.0`
|}

== Compatibility Notes ==

* '''Apple Silicon (M1/M2):''' Requires `tensorflow-macos` instead of standard `tensorflow`. The requirements.txt handles this automatically with platform markers.
* '''Windows:''' Fully supported with standard TensorFlow.
* '''Linux:''' Fully supported with standard TensorFlow.
* '''Model sizes:''' Larger models (774M, 1558M) require more RAM and disk space. The 1558M model checkpoint is approximately 6GB.
* '''Offline usage:''' Models are downloaded once to the `models/` directory and cached for subsequent runs.

== Related Pages ==
* [[required_by::Implementation:Jaymody_PicoGPT_Download_Gpt2_Files]]
* [[required_by::Implementation:Jaymody_PicoGPT_Load_Gpt2_Params_From_Tf_Ckpt]]
* [[required_by::Implementation:Jaymody_PicoGPT_Encoder]]
* [[required_by::Implementation:Jaymody_PicoGPT_Encoder_Encode]]
* [[required_by::Implementation:Jaymody_PicoGPT_Encoder_Decode]]
* [[required_by::Implementation:Jaymody_PicoGPT_Gpt2]]
* [[required_by::Implementation:Jaymody_PicoGPT_Generate]]
