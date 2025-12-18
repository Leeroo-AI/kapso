# Heuristic: huggingface_transformers_Device_Map_Strategy

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Big Model Inference|https://huggingface.co/docs/accelerate/usage_guides/big_modeling]]
|-
! Domains
| [[domain::Device_Placement]], [[domain::Memory]], [[domain::Multi_GPU]]
|-
! Last Updated
| [[last_updated::2025-12-18 14:00 GMT]]
|}

== Overview ==
Use `device_map="auto"` for automatic model distribution across GPUs, with manual override for specific placement requirements.

=== Description ===
Device mapping controls how model layers are distributed across available devices (GPUs, CPU, disk). The `"auto"` strategy uses accelerate's intelligent placement algorithm that considers available memory on each device. For multi-GPU setups, this enables running models larger than a single GPU's VRAM by splitting layers across devices.

=== Usage ===
Use `device_map="auto"` when loading models larger than your GPU VRAM, or when you want automatic multi-GPU distribution. Use manual device maps when you need specific control over layer placement or want to reserve GPU memory for other operations.

== The Insight (Rule of Thumb) ==

* **"auto":** Let accelerate handle placement automatically (recommended default)
* **"balanced":** Distribute layers evenly across GPUs (may not be optimal)
* **"sequential":** Fill GPUs in order (first GPU gets layers until full)
* **Manual dict:** Explicit control: `{"model.embed_tokens": 0, "model.layers.0": 0, ...}`

== Reasoning ==

Automatic device mapping considers:
1. Available GPU memory on each device
2. Layer sizes and activation memory requirements
3. Communication overhead between devices
4. CPU offloading as last resort

Manual mapping is useful when:
- You need deterministic placement
- Some GPUs are shared with other processes
- You want to pin specific layers (like lm_head) to specific devices

== Code Evidence ==

From `modeling_utils.py`:

<syntaxhighlight lang="python">
if device_map is not None:
    if not is_accelerate_available():
        raise ImportError(
            "Using `device_map` requires the accelerate library: "
            "`pip install 'accelerate>=1.1.0'`"
        )
</syntaxhighlight>

From `quantizer_bnb_8bit.py:L86-103`:

<syntaxhighlight lang="python">
def update_device_map(self, device_map):
    if device_map is None:
        if torch.cuda.is_available():
            device_map = {"": torch.cuda.current_device()}
        elif is_torch_npu_available():
            device_map = {"": f"npu:{torch.npu.current_device()}"}
        elif is_torch_xpu_available():
            device_map = {"": torch.xpu.current_device()}
        else:
            device_map = {"": "cpu"}
    return device_map
</syntaxhighlight>

== Example Usage ==

<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM

# Automatic placement (recommended)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    device_map="auto",
    torch_dtype=torch.float16,
)

# With memory limits per GPU
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    device_map="auto",
    max_memory={0: "20GiB", 1: "20GiB", "cpu": "30GiB"},
)

# Single GPU placement
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    device_map={"": 0},  # All on GPU 0
)

# Manual layer placement (advanced)
device_map = {
    "model.embed_tokens": 0,
    "model.layers.0": 0,
    "model.layers.1": 0,
    # ... layers 2-15 on GPU 0
    "model.layers.16": 1,
    # ... layers 16-31 on GPU 1
    "model.norm": 1,
    "lm_head": 1,
}
</syntaxhighlight>

== Device Map Options ==

{| class="wikitable"
|-
! Option !! Description !! Use Case
|-
| `"auto"` || Automatic optimal placement || Default for most cases
|-
| `"balanced"` || Even distribution across GPUs || When all GPUs are equal
|-
| `"balanced_low_0"` || Balanced but minimize GPU 0 usage || Keep GPU 0 free for inference batching
|-
| `"sequential"` || Fill GPUs in order || Predictable placement
|-
| `{"": 0}` || All on single GPU || Single GPU inference
|-
| `{"": "cpu"}` || All on CPU || CPU-only inference
|-
| Manual dict || Explicit layer-device mapping || Fine-grained control
|}

== Memory Management ==

<syntaxhighlight lang="python">
# Set max memory per device
max_memory = {
    0: "22GiB",      # GPU 0
    1: "22GiB",      # GPU 1
    "cpu": "50GiB",  # CPU RAM (for offloading)
}

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    max_memory=max_memory,
    offload_folder="./offload",  # For disk offload
)
</syntaxhighlight>

== Related Pages ==
* [[uses_heuristic::Implementation:huggingface_transformers_Accelerate_dispatch]]
* [[uses_heuristic::Implementation:huggingface_transformers_Model_initialization]]
* [[uses_heuristic::Workflow:huggingface_transformers_Model_Loading]]
