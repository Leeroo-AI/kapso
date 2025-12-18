# Heuristic: huggingface_transformers_Safetensors_Preference

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
* [[source::Repo|Safetensors|https://github.com/huggingface/safetensors]]
|-
! Domains
| [[domain::Serialization]], [[domain::Security]], [[domain::Model_Loading]]
|-
! Last Updated
| [[last_updated::2025-12-18 14:00 GMT]]
|}

== Overview ==
Always prefer safetensors format over .bin/.pt files for faster, safer, and more memory-efficient model loading.

=== Description ===
Safetensors is a safe and fast file format for storing tensors, designed as a secure alternative to pickle-based formats (.pt, .bin). It provides memory-mapped loading for efficient random access, is resistant to arbitrary code execution, and is typically 2-4x faster to load than pickle formats.

=== Usage ===
Use safetensors format for all model storage and distribution. When loading models, prefer `safe_serialization=True` when saving and look for `.safetensors` files when downloading. Most HuggingFace Hub models now default to safetensors.

== The Insight (Rule of Thumb) ==

* **Action:** Save with `model.save_pretrained(..., safe_serialization=True)` (default since v4.37)
* **Value:** 2-4x faster loading, memory-mapped random access, no pickle vulnerabilities
* **Trade-off:** None; safetensors is strictly better for well-formed models
* **Security:** Eliminates arbitrary code execution risk from malicious model files

== Reasoning ==

Pickle format issues:
1. Arbitrary code execution during `torch.load()` (security risk)
2. Requires loading entire file into memory
3. No random access to individual tensors
4. Complex format prone to version incompatibilities

Safetensors advantages:
1. No code execution possible (safe by design)
2. Memory-mapped: only loads tensors when accessed
3. Random access: can load individual layers without full file read
4. Simple format: header + flat tensor data
5. Cross-platform compatibility

== Code Evidence ==

From `modeling_utils.py`:

<syntaxhighlight lang="python">
# Safetensors is preferred by default
use_safetensors = resolved_archive_file.endswith(".safetensors")
if use_safetensors:
    from safetensors.torch import load_file as safe_load_file
    state_dict = safe_load_file(resolved_archive_file)
</syntaxhighlight>

Default save format from `modeling_utils.py`:

<syntaxhighlight lang="python">
def save_pretrained(
    self,
    save_directory,
    safe_serialization=True,  # Default since v4.37
    ...
):
</syntaxhighlight>

Version requirement from `dependency_versions_table.py`:

<syntaxhighlight lang="python">
deps = {
    "safetensors": "safetensors>=0.4.3",
}
</syntaxhighlight>

== Example Usage ==

<syntaxhighlight lang="python">
from transformers import AutoModel, AutoModelForCausalLM

# Loading (automatic safetensors preference)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    # Safetensors automatically used if available
)

# Explicit safetensors request
model = AutoModel.from_pretrained(
    "bert-base-uncased",
    use_safetensors=True,  # Fail if only .bin available
)

# Saving in safetensors format (default)
model.save_pretrained(
    "./my_model",
    safe_serialization=True,  # Default, can be omitted
)

# Force pickle format (not recommended)
model.save_pretrained(
    "./my_model_pickle",
    safe_serialization=False,
)
</syntaxhighlight>

== Format Comparison ==

{| class="wikitable"
|-
! Aspect !! Safetensors !! Pickle (.pt/.bin)
|-
| Load speed || Fast (2-4x) || Baseline
|-
| Memory mapping || Yes || No
|-
| Random access || Yes || No
|-
| Security || Safe || Code execution risk
|-
| Cross-platform || Excellent || Good
|-
| Hub default || Yes (since 2023) || Legacy
|}

== File Detection ==

<syntaxhighlight lang="python">
import os

def get_checkpoint_format(model_path):
    """Detect checkpoint format."""
    safetensors_files = [f for f in os.listdir(model_path)
                        if f.endswith(".safetensors")]
    bin_files = [f for f in os.listdir(model_path)
                if f.endswith(".bin") or f.endswith(".pt")]

    if safetensors_files:
        return "safetensors"
    elif bin_files:
        return "pickle"
    else:
        return "unknown"
</syntaxhighlight>

== Related Pages ==
* [[uses_heuristic::Implementation:huggingface_transformers_Weight_loading]]
* [[uses_heuristic::Implementation:huggingface_transformers_Model_saving]]
* [[uses_heuristic::Workflow:huggingface_transformers_Model_Loading]]
