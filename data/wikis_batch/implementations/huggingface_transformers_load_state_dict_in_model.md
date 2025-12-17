{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Transformers Docs|https://huggingface.co/docs/transformers]]
|-
! Domains
| [[domain::NLP]], [[domain::Model_Loading]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

== Overview ==

Concrete tool for loading and converting checkpoint weights into a model's parameter structure with support for quantization, device mapping, and weight transformations provided by the HuggingFace Transformers library.

=== Description ===

`convert_and_load_state_dict_in_model()` is the core weight loading function that bridges checkpoint state dictionaries and live model parameters. It handles the complexity of modern loading scenarios: parameter renaming between checkpoint and model conventions, weight conversions (e.g., splitting QKV attention weights), quantization (loading full-precision weights into quantized parameters), device placement (distributing weights across GPUs/CPU/disk according to device map), dtype casting, and tensor parallelism. The function operates on a model with potentially meta-device parameters, materializing them with actual weight data while applying necessary transformations.

=== Usage ===

Use this when you need to:
* Load checkpoint weights into a model instantiated with init_empty_weights
* Apply weight transformations during loading (merging, splitting, reshaping)
* Load weights with automatic device placement and quantization
* Implement custom model loading pipelines with fine-grained control
* Handle renamed parameters between checkpoint formats and model code

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' src/transformers/core_model_loading.py (lines 716-800+)

=== Signature ===
<syntaxhighlight lang="python">
def convert_and_load_state_dict_in_model(
    model: PreTrainedModel,
    state_dict: dict[str, Any],
    weight_mapping: list[WeightConverter | WeightRenaming] | None,
    tp_plan: dict[str, str] | None,
    hf_quantizer: HfQuantizer | None,
    dtype: torch.dtype | None = None,
    device_map: dict | None = None,
    dtype_plan: dict | None = None,
    device_mesh: torch.distributed.device_mesh.DeviceMesh | None = None,
    disk_offload_index: dict | None = None,
    disk_offload_folder: str | None = None,
):
    """
    Load checkpoint weights into model, applying conversions, quantization, and device placement.

    Args:
        model (PreTrainedModel): Target model with potentially uninitialized (meta) parameters
        state_dict (dict[str, Any]): Checkpoint state dictionary mapping parameter names to tensors
        weight_mapping (list[WeightConverter | WeightRenaming] | None): Rules for renaming
            and transforming weights (e.g., split QKV, merge experts)
        tp_plan (dict[str, str] | None): Tensor parallelism plan mapping parameter names
            to sharding strategies
        hf_quantizer (HfQuantizer | None): Quantizer instance for loading into quantized format
        dtype (torch.dtype | None): Target dtype for weights (e.g., torch.float16)
        device_map (dict | None): Device placement mapping {layer_name: device}
        dtype_plan (dict | None): Per-parameter dtype overrides
        device_mesh (DeviceMesh | None): Distributed device mesh for tensor parallel loading
        disk_offload_index (dict | None): Index of disk-offloaded parameters
        disk_offload_folder (str | None): Directory for disk offloading

    Returns:
        None (modifies model in-place)

    Side Effects:
        - model parameters are materialized from meta device with actual weight data
        - model parameters placed on target devices according to device_map
        - quantization applied if hf_quantizer provided
        - weight conversions applied according to weight_mapping
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from transformers.core_model_loading import convert_and_load_state_dict_in_model
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model || PreTrainedModel || Yes || Model instance with architecture instantiated, parameters potentially on meta device
|-
| state_dict || dict[str, Any] || Yes || Checkpoint state dictionary mapping parameter names to weight tensors
|-
| weight_mapping || list[WeightConverter | WeightRenaming] | None || No || List of weight transformation rules for renaming or converting parameters
|-
| tp_plan || dict[str, str] | None || No || Tensor parallelism plan specifying sharding strategy per parameter
|-
| hf_quantizer || HfQuantizer | None || No || Quantizer instance for loading weights in quantized format
|-
| dtype || torch.dtype | None || No || Target data type for weights (e.g., torch.float16, torch.bfloat16)
|-
| device_map || dict | None || No || Device placement mapping specifying which device each parameter/layer should be loaded to
|-
| dtype_plan || dict | None || No || Per-parameter dtype overrides for mixed precision loading
|-
| device_mesh || DeviceMesh | None || No || Distributed device mesh for tensor parallel loading across multiple devices
|-
| disk_offload_index || dict | None || No || Index of disk-offloaded parameters for CPU/disk offloading
|-
| disk_offload_folder || str | None || No || Directory path for disk offloading large parameters
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| None || None || Function modifies model in-place; no return value
|}

'''Post-Execution State:'''
* Model parameters materialized with checkpoint weights
* Parameters placed on target devices per device_map
* Quantization applied if quantizer provided
* Weight conversions executed (rename, split, merge operations)
* Missing keys and unexpected keys tracked for diagnostics

== Usage Examples ==

=== Basic Usage ===
<syntaxhighlight lang="python">
from transformers import AutoModel, AutoConfig
from transformers.core_model_loading import convert_and_load_state_dict_in_model
from accelerate import init_empty_weights
import torch
from safetensors.torch import load_file

# Load configuration and instantiate empty model
config = AutoConfig.from_pretrained("google-bert/bert-base-uncased")
with init_empty_weights():
    model = AutoModel.from_config(config)

# Load checkpoint state dict
state_dict = load_file("path/to/model.safetensors")

# Simple loading: no conversions, single device
convert_and_load_state_dict_in_model(
    model=model,
    state_dict=state_dict,
    weight_mapping=None,
    tp_plan=None,
    hf_quantizer=None,
    dtype=torch.float32,
    device_map={"": "cuda:0"}
)

print(f"Model loaded on {next(model.parameters()).device}")

# With dtype conversion
config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf")
with init_empty_weights():
    model = AutoModel.from_config(config)

state_dict = load_file("path/to/llama/model.safetensors")

convert_and_load_state_dict_in_model(
    model=model,
    state_dict=state_dict,
    weight_mapping=None,
    tp_plan=None,
    hf_quantizer=None,
    dtype=torch.float16,  # Convert to half precision
    device_map={"": "cuda:0"}
)

# Multi-device loading with device map
device_map = {
    "model.embed_tokens": "cuda:0",
    "model.layers.0": "cuda:0",
    "model.layers.1": "cuda:0",
    # ... more layers
    "model.layers.30": "cuda:1",
    "model.layers.31": "cuda:1",
    "lm_head": "cuda:1"
}

convert_and_load_state_dict_in_model(
    model=model,
    state_dict=state_dict,
    weight_mapping=None,
    tp_plan=None,
    hf_quantizer=None,
    dtype=torch.float16,
    device_map=device_map
)

# With quantization
from transformers.quantizers.auto import get_hf_quantizer
from transformers.utils.quantization_config import BitsAndBytesConfig

config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf")
bnb_config = BitsAndBytesConfig(load_in_4bit=True)

hf_quantizer = get_hf_quantizer(
    config=config,
    quantization_config=bnb_config,
    device_map="auto",
    weights_only=False,
    user_agent={}
)

with init_empty_weights():
    model = AutoModel.from_config(config)

state_dict = load_file("path/to/llama/model.safetensors")

convert_and_load_state_dict_in_model(
    model=model,
    state_dict=state_dict,
    weight_mapping=None,
    tp_plan=None,
    hf_quantizer=hf_quantizer,  # Weights loaded as 4-bit quantized
    dtype=None,  # Quantizer determines dtype
    device_map={"": "cuda:0"}
)

# With weight conversions (e.g., QKV splitting)
from transformers.core_model_loading import WeightConverter, Chunk

# Example: checkpoint has fused QKV weights, model expects separate Q, K, V
weight_mapping = [
    WeightConverter(
        source_patterns=["attention.qkv.weight"],
        target_patterns=["attention.q.weight", "attention.k.weight", "attention.v.weight"],
        operations=[Chunk(dim=0, chunks=3)]
    )
]

convert_and_load_state_dict_in_model(
    model=model,
    state_dict=state_dict,
    weight_mapping=weight_mapping,  # Apply conversions
    tp_plan=None,
    hf_quantizer=None,
    dtype=torch.float16,
    device_map={"": "cuda:0"}
)

# Disk offloading for very large models
disk_offload_folder = "./offload"
os.makedirs(disk_offload_folder, exist_ok=True)

convert_and_load_state_dict_in_model(
    model=model,
    state_dict=state_dict,
    weight_mapping=None,
    tp_plan=None,
    hf_quantizer=None,
    dtype=torch.float16,
    device_map={
        "model.embed_tokens": "cuda:0",
        "model.layers.0-10": "cuda:0",
        "model.layers.11-20": "cpu",  # CPU offload
        "model.layers.21-31": "disk",  # Disk offload
        "lm_head": "cuda:0"
    },
    disk_offload_folder=disk_offload_folder
)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Weight_Loading]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_transformers_PyTorch]]
