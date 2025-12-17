# Environment: huggingface_transformers_PyTorch

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Transformers Installation|https://huggingface.co/docs/transformers/installation]]
|-
! Domains
| [[domain::Infrastructure]], [[domain::Deep_Learning]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

== Overview ==

Base Python environment with PyTorch 2.2+, Accelerate, and core dependencies for running HuggingFace Transformers.

=== Description ===

This environment provides the foundational runtime context for all HuggingFace Transformers operations. It includes the PyTorch deep learning framework as the primary backend, along with essential utilities like Accelerate for device management, Tokenizers for fast text processing, and SafeTensors for efficient model serialization. This is the minimum required environment for model loading, inference, and basic training workflows.

=== Usage ===

Use this environment for any HuggingFace Transformers operation including:
- **Model Loading**: `AutoModel.from_pretrained()`
- **Tokenization**: `AutoTokenizer.from_pretrained()`
- **Basic Training**: `Trainer` class (CPU or single GPU)
- **Pipeline Inference**: `pipeline()` factory function

This is the mandatory prerequisite for all other specialized environments (CUDA, BitsAndBytes, Flash Attention).

== System Requirements ==

{| class="wikitable"
! Category !! Requirement !! Notes
|-
| OS || Linux, macOS, Windows || Linux recommended for production
|-
| Python || >= 3.10.0 || Required (validated at runtime)
|-
| Hardware || CPU || GPU optional for this base environment
|-
| Disk || 10GB+ || For model caching (~/.cache/huggingface)
|}

== Dependencies ==

=== Core Dependencies (Required) ===

* `torch` >= 2.2
* `accelerate` >= 1.1.0
* `tokenizers` >= 0.22.0, <= 0.23.0
* `safetensors` >= 0.4.3
* `huggingface-hub` >= 1.2.1, < 2.0

=== System Packages (Required) ===

* `tqdm` >= 4.27
* `regex` != 2019.12.17
* `requests`
* `filelock`
* `numpy` >= 1.17
* `packaging` >= 20.0
* `pyyaml` >= 5.1

=== Optional Dependencies ===

* `datasets` >= 2.15.0 (for dataset loading)
* `peft` >= 0.18.0 (for parameter-efficient fine-tuning)
* `jinja2` >= 3.1.0 (for chat templates)
* `sentencepiece` >= 0.1.91, != 0.1.92 (for some tokenizers)

== Credentials ==

The following environment variables may be required:

* `HF_TOKEN`: HuggingFace API token for accessing gated models and private repositories
* `HF_HOME`: Custom cache directory (default: `~/.cache/huggingface`)
* `TRANSFORMERS_CACHE`: Legacy cache directory override

== Quick Install ==

<syntaxhighlight lang="bash">
# Install core transformers with PyTorch
pip install transformers[torch] accelerate

# Or install components individually
pip install torch>=2.2 transformers accelerate safetensors tokenizers

# For full feature set
pip install transformers[torch,sentencepiece,tokenizers] accelerate datasets
</syntaxhighlight>

== Code Evidence ==

Runtime dependency validation from `dependency_versions_check.py:25-37`:

<syntaxhighlight lang="python">
pkgs_to_check_at_runtime = [
    "python",
    "tqdm",
    "regex",
    "requests",
    "packaging",
    "filelock",
    "numpy",
    "tokenizers",
    "huggingface-hub",
    "safetensors",
    "accelerate",
    "pyyaml",
]
</syntaxhighlight>

Version constraints from `dependency_versions_table.py`:

<syntaxhighlight lang="python">
deps = {
    "torch": "torch>=2.2",
    "accelerate": "accelerate>=1.1.0",
    "safetensors": "safetensors>=0.4.3",
    "tokenizers": "tokenizers>=0.22.0,<=0.23.0",
    "huggingface-hub": "huggingface-hub>=1.2.1,<2.0",
    "python": "python>=3.10.0",
    # ...
}
</syntaxhighlight>

Accelerate availability check from `trainer.py:211`:

<syntaxhighlight lang="python">
if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches
    from accelerate import __version__ as accelerate_version
</syntaxhighlight>

== Common Errors ==

{| class="wikitable"
|-
! Error Message !! Cause !! Solution
|-
|| `ImportError: No module named 'torch'` || PyTorch not installed || `pip install torch>=2.2`
|-
|| `Version mismatch: tokenizers` || Tokenizers version outside allowed range || `pip install tokenizers>=0.22.0,<=0.23.0`
|-
|| `ImportError: accelerate required` || Accelerate not installed || `pip install accelerate>=1.1.0`
|-
|| `OSError: Can't load tokenizer` || Missing tokenizer files || Check `HF_TOKEN` for gated models
|}

== Compatibility Notes ==

* **Windows**: Full support via PyTorch Windows builds
* **macOS**: CPU support; MPS (Metal) acceleration available on Apple Silicon
* **Linux**: Full support; recommended for training workloads
* **Python 3.9**: Not supported as of recent versions (3.10+ required)

== Related Pages ==

* [[requires_env::Implementation:huggingface_transformers_AutoConfig_from_pretrained]]
* [[requires_env::Implementation:huggingface_transformers_AutoTokenizer_from_pretrained]]
* [[requires_env::Implementation:huggingface_transformers_Trainer_init]]
* [[requires_env::Implementation:huggingface_transformers_pipeline_factory]]
