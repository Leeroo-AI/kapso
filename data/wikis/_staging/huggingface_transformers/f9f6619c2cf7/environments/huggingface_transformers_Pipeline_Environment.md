# Environment: huggingface_transformers_Pipeline_Environment

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Pipeline Documentation|https://huggingface.co/docs/transformers/main_classes/pipelines]]
|-
! Domains
| [[domain::Inference]], [[domain::NLP]], [[domain::Computer_Vision]], [[domain::Audio]]
|-
! Last Updated
| [[last_updated::2025-12-18 14:00 GMT]]
|}

== Overview ==
Python 3.10+ environment with PyTorch 2.2+, tokenizers, and huggingface-hub for running inference pipelines.

=== Description ===
This environment provides the base context for running HuggingFace Transformers pipelines. It includes the core dependencies required for loading models from the Hub, processing inputs with tokenizers/processors, and running inference. The environment supports CPU inference by default, with optional GPU acceleration when CUDA or other accelerators are available.

=== Usage ===
Use this environment for any **inference** workflow using the `pipeline()` factory function. Required for `text-generation`, `text-classification`, `question-answering`, `image-classification`, `automatic-speech-recognition`, and all other pipeline tasks.

== System Requirements ==

{| class="wikitable"
! Category !! Requirement !! Notes
|-
| OS || Linux, Windows, macOS || All major platforms supported
|-
| Python || Python >= 3.10 || Required by transformers
|-
| Hardware || CPU or GPU || GPU optional, improves performance
|-
| Memory || 4GB+ RAM || Depends on model size
|}

== Dependencies ==

=== System Packages ===
* None required for basic inference

=== Python Packages ===
* `transformers` (this package)
* `torch` >= 2.2
* `tokenizers` >= 0.22.0, <= 0.23.0
* `huggingface-hub` >= 1.2.1, < 2.0
* `safetensors` >= 0.4.3
* `numpy` >= 1.17
* `tqdm` >= 4.27
* `packaging` >= 20.0
* `filelock`
* `requests`
* `regex` (not 2019.12.17)
* `pyyaml` >= 5.1

=== Optional Dependencies ===
* `accelerate` >= 1.1.0 - For device_map and multi-GPU support
* `Pillow` >= 10.0.1, <= 15.0 - For image pipelines
* `librosa` - For audio pipelines
* `sentencepiece` >= 0.1.91 - For some tokenizers

== Credentials ==
The following environment variables are optional:
* `HF_TOKEN`: HuggingFace API token for accessing gated models
* `HF_HOME`: Custom cache directory (default: `~/.cache/huggingface`)

== Quick Install ==

<syntaxhighlight lang="bash">
# Basic installation
pip install transformers torch tokenizers huggingface-hub safetensors

# With image processing
pip install transformers[vision] torch

# With audio processing
pip install transformers[audio] torch
</syntaxhighlight>

== Code Evidence ==

Dependency version checks from `dependency_versions_table.py`:

<syntaxhighlight lang="python">
deps = {
    "torch": "torch>=2.2",
    "tokenizers": "tokenizers>=0.22.0,<=0.23.0",
    "huggingface-hub": "huggingface-hub>=1.2.1,<2.0",
    "safetensors": "safetensors>=0.4.3",
    "numpy": "numpy>=1.17",
    "python": "python>=3.10.0",
}
</syntaxhighlight>

Device auto-detection from `pipelines/base.py`:

<syntaxhighlight lang="python">
if device is None:
    if is_torch_cuda_available():
        device = 0  # Default to first GPU
    elif is_torch_mps_available():
        device = "mps"
    else:
        device = -1  # CPU
</syntaxhighlight>

== Common Errors ==

{| class="wikitable"
|-
! Error Message !! Cause !! Solution
|-
|| `ModuleNotFoundError: No module named 'torch'` || PyTorch not installed || `pip install torch`
|-
|| `ImportError: tokenizers>=0.22.0 is required` || Tokenizers version too old || `pip install -U tokenizers`
|-
|| `OSError: ... is a gated model` || Access token not set || Set `HF_TOKEN` or use `token=` parameter
|-
|| `RuntimeError: CUDA out of memory` || Insufficient GPU VRAM || Use `device_map="auto"` or smaller model
|}

== Compatibility Notes ==

* **CPU:** Works on all platforms without additional dependencies
* **CUDA:** Requires NVIDIA GPU with PyTorch CUDA support
* **MPS:** Supported on Apple Silicon Macs with PyTorch >= 2.0
* **XPU:** Intel GPU support available with PyTorch XPU extensions

== Related Pages ==
* [[requires_env::Implementation:huggingface_transformers_Pipeline_factory_function]]
* [[requires_env::Implementation:huggingface_transformers_AutoProcessor_initialization]]
* [[requires_env::Implementation:huggingface_transformers_Pipeline_model_initialization]]
* [[requires_env::Implementation:huggingface_transformers_Pipeline_preprocess]]
* [[requires_env::Implementation:huggingface_transformers_Pipeline_forward_pass]]
* [[requires_env::Implementation:huggingface_transformers_Pipeline_postprocess]]
