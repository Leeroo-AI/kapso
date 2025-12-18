# Environment: huggingface_transformers_Loading_Environment

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Model Loading Documentation|https://huggingface.co/docs/transformers/main_classes/model]]
|-
! Domains
| [[domain::Model_Loading]], [[domain::Infrastructure]], [[domain::Deep_Learning]]
|-
! Last Updated
| [[last_updated::2025-12-18 14:00 GMT]]
|}

== Overview ==
Environment for loading pretrained models from Hub or local checkpoints with device mapping and optional quantization support.

=== Description ===
This environment provides the context for loading Transformer models using `from_pretrained()`. It supports loading from HuggingFace Hub, local directories, or URLs. The environment handles device placement (CPU, GPU, multi-GPU with device_map), weight format conversion (safetensors, bin), and optional quantization (bitsandbytes, GPTQ, AWQ).

=== Usage ===
Use this environment when loading models with `AutoModel.from_pretrained()`, `AutoModelForCausalLM.from_pretrained()`, or any model's `from_pretrained()` method. Required for all workflows that need to load pretrained weights.

== System Requirements ==

{| class="wikitable"
! Category !! Requirement !! Notes
|-
| OS || Linux, Windows, macOS || All platforms supported
|-
| Python || Python >= 3.10 || Required by transformers
|-
| Hardware || CPU or GPU || Large models may require GPU
|-
| Memory || Model-dependent || 2x model size for fp16, 1x for 8-bit
|-
| Disk || 10GB+ || For model cache
|}

== Dependencies ==

=== System Packages ===
* None required for basic loading

=== Python Packages ===
* `transformers` (this package)
* `torch` >= 2.2
* `safetensors` >= 0.4.3 - For loading safetensors format
* `huggingface-hub` >= 1.2.1 - For Hub downloads
* `tokenizers` >= 0.22.0

=== Optional Dependencies ===
* `accelerate` >= 1.1.0 - For device_map support
* `bitsandbytes` - For 4-bit/8-bit quantization
* `auto-gptq` - For GPTQ quantized models
* `autoawq` - For AWQ quantized models
* `optimum` - For additional quantization methods

== Credentials ==
The following environment variables are optional:
* `HF_TOKEN`: HuggingFace API token for gated/private models
* `HF_HOME`: Custom cache directory (default: `~/.cache/huggingface`)
* `TRANSFORMERS_CACHE`: Legacy cache directory (deprecated)

== Quick Install ==

<syntaxhighlight lang="bash">
# Basic model loading
pip install transformers torch safetensors huggingface-hub

# With device_map support (multi-GPU/offloading)
pip install transformers torch safetensors huggingface-hub accelerate

# With quantization support
pip install transformers torch safetensors accelerate bitsandbytes
</syntaxhighlight>

== Code Evidence ==

Safetensors preference from `modeling_utils.py`:

<syntaxhighlight lang="python">
# Safetensors is preferred by default
use_safetensors = resolved_archive_file.endswith(".safetensors")
if use_safetensors:
    from safetensors.torch import load_file as safe_load_file
    state_dict = safe_load_file(resolved_archive_file)
</syntaxhighlight>

Device map requirement from `modeling_utils.py`:

<syntaxhighlight lang="python">
if device_map is not None:
    if not is_accelerate_available():
        raise ImportError(
            "Using `device_map` requires the `accelerate` library: "
            "`pip install 'accelerate>=1.1.0'`"
        )
</syntaxhighlight>

Weight loading from `modeling_utils.py:L317-349`:

<syntaxhighlight lang="python">
def load_state_dict(checkpoint_file, map_location="cpu", weights_only=True):
    """Load a state dict from a checkpoint file."""
    if checkpoint_file.endswith(".safetensors"):
        return safe_load_file(checkpoint_file, device=map_location)
    return torch.load(checkpoint_file, map_location=map_location, weights_only=weights_only)
</syntaxhighlight>

== Common Errors ==

{| class="wikitable"
|-
! Error Message !! Cause !! Solution
|-
|| `OSError: ... is a gated model` || Model requires authentication || `huggingface-cli login` or set `HF_TOKEN`
|-
|| `ImportError: accelerate is required for device_map` || Accelerate not installed || `pip install accelerate`
|-
|| `OutOfMemoryError: CUDA out of memory` || Model too large for GPU || Use `device_map="auto"` or quantization
|-
|| `RuntimeError: Expected all tensors to be on the same device` || Device mismatch || Use `device_map` for multi-device models
|-
|| `ValueError: Unknown quantization type` || Unsupported quantization || Check supported methods in `quantizers/auto.py`
|}

== Compatibility Notes ==

* **Safetensors vs .bin:** Safetensors is faster and safer; preferred by default since v4.37
* **Device Map "auto":** Requires accelerate; automatically distributes model across available GPUs
* **CPU Offloading:** Use `device_map="auto"` with `max_memory` to offload to CPU/disk
* **Quantization:** Different backends have different GPU requirements (e.g., bitsandbytes requires CUDA)
* **Remote Code:** Use `trust_remote_code=True` for models with custom code (security implications)

== Related Pages ==
* [[requires_env::Implementation:huggingface_transformers_PretrainedConfig_from_pretrained]]
* [[requires_env::Implementation:huggingface_transformers_Checkpoint_file_resolution]]
* [[requires_env::Implementation:huggingface_transformers_Quantizer_setup]]
* [[requires_env::Implementation:huggingface_transformers_Model_initialization]]
* [[requires_env::Implementation:huggingface_transformers_Weight_loading]]
* [[requires_env::Implementation:huggingface_transformers_Accelerate_dispatch]]
* [[requires_env::Implementation:huggingface_transformers_Post_init_processing]]
