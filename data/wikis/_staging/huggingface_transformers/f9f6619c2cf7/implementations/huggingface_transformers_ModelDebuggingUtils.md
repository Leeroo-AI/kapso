{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|huggingface_transformers|https://github.com/huggingface/transformers]]
|-
! Domains
| [[domain::Debugging]], [[domain::Model Development]], [[domain::Utilities]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
A context manager that traces the complete forward pass of a PyTorch model, recording inputs/outputs of every module to structured JSON files for debugging and verification during model development.

=== Description ===
The `model_addition_debugger_context` is a power-user tool designed specifically for developers adding new models to the Transformers library. It wraps a model's forward pass to capture detailed information about every submodule call, including tensor shapes, dtypes, statistics (mean, std, min, max), and optionally full tensor values. The trace is saved as nested JSON representing the model's call tree, with options to save either tensor representations or full tensors in SafeTensors format.

This tool helps model developers verify correct implementation, debug shape mismatches, track value distributions through layers, and create reproducible test cases. It automatically handles distributed training (only rank 0 saves data) and DTensor objects, making it suitable for modern distributed training setups.

=== Usage ===
Use this context manager when adding new model architectures to Transformers, debugging unexpected model behaviors, verifying correctness against reference implementations, or creating detailed documentation of model internals. It's particularly valuable for comparing outputs between original implementations and HuggingFace ports, as the JSON traces enable precise diff analysis.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers huggingface_transformers]
* '''File:''' src/transformers/model_debugging_utils.py

=== Signature ===
<syntaxhighlight lang="python">
@requires(backends=("torch",))
@contextmanager
def model_addition_debugger_context(
    model,
    debug_path: Optional[str] = None,
    do_prune_layers: bool = True,
    use_repr: bool = True
):
    """
    Context manager for tracing model forward passes.

    Args:
        model: The model to debug (PreTrainedModel or nn.Module)
        debug_path: Directory to save JSON/SafeTensors files (default: current dir)
        do_prune_layers: Whether to prune intermediate layers (keep first/last)
        use_repr: True for tensor repr in JSON, False for SafeTensors files

    Yields:
        The model with debugging hooks attached
    """

# Helper functions
def _serialize_io(value, debug_path, use_repr, path_to_value):
    """Recursively serialize tensors/lists/dicts to JSON-safe structure."""

def _serialize_tensor_like_io(value, debug_path, use_repr, path_to_value):
    """Convert single tensor to JSON-safe dict with shape/dtype/stats."""

def prune_outputs_if_children(node):
    """Remove outputs from parent nodes, keeping only leaf outputs."""

def prune_intermediate_layers(node):
    """Remove intermediate layers from trace, keeping first and last."""

def log_model_debug_trace(debug_path, model):
    """Write call tree to JSON files (_FULL_TENSORS.json and _SUMMARY.json)."""

def _attach_debugger_logic(model, debug_path, do_prune_layers, use_repr):
    """Wrap all module forward methods with tracing logic."""

def _is_rank_zero():
    """Check if current process is rank 0 in distributed setting."""

def _sanitize_repr_for_diff(x_str):
    """Replace memory addresses with stable placeholder for clean diffs."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from transformers import model_addition_debugger_context
# or
from transformers.model_debugging_utils import model_addition_debugger_context
</syntaxhighlight>

== I/O Contract ==

=== Context Manager Parameters ===
{| class="wikitable"
! Parameter !! Type !! Default !! Description
|-
| model || nn.Module || Required || The model to trace (must be PyTorch)
|-
| debug_path || str || None || Directory for output files (uses cwd if None)
|-
| do_prune_layers || bool || True || Remove intermediate layer blocks from trace
|-
| use_repr || bool || True || Save tensor repr strings vs. SafeTensors files
|}

=== Output Files ===
{| class="wikitable"
! File Pattern !! Description
|-
| {ModelName}_debug_tree_FULL_TENSORS.json || Complete trace with tensor values/reprs
|-
| {ModelName}_debug_tree_SUMMARY.json || Trace with stats only (no tensor values)
|-
| {path}_inputs.safetensors || Input tensor data (if use_repr=False)
|-
| {path}_outputs.safetensors || Output tensor data (if use_repr=False)
|}

=== JSON Structure ===
Each node in the call tree contains:
{| class="wikitable"
! Field !! Type !! Description
|-
| module_path || str || Fully qualified module path (e.g., "Model.encoder.layer.0")
|-
| inputs || dict || {"args": [...], "kwargs": {...}} with serialized tensors
|-
| outputs || dict/list || Serialized output tensors (None for parent nodes)
|-
| children || list || Nested list of child module calls
|}

=== Tensor Serialization Format ===
Each tensor becomes:
{| class="wikitable"
! Field !! Type !! Description
|-
| shape || str || Tensor shape repr (e.g., "torch.Size([32, 768])")
|-
| dtype || str || Data type (e.g., "torch.float32")
|-
| value || list[str] or str || Repr lines if use_repr=True, else path to .safetensors
|-
| mean || str || Mean value (float dtypes only)
|-
| std || str || Standard deviation (float dtypes only)
|-
| min || str || Minimum value (float dtypes only)
|-
| max || str || Maximum value (float dtypes only)
|}

== Usage Examples ==

=== Basic Usage with Vision-Language Model ===
<syntaxhighlight lang="python">
import torch
from PIL import Image
from transformers import (
    LlavaProcessor,
    LlavaForConditionalGeneration,
    model_addition_debugger_context
)

# Set seed for reproducibility
torch.random.manual_seed(673)

# Load model and processor
model_id = "llava-hf/llava-1.5-7b-hf"
processor = LlavaProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(model_id)

# Create input
random_image = Image.fromarray(
    torch.randint(0, 256, (224, 224, 3), dtype=torch.uint8).numpy()
)
prompt = "<image>Describe this image."
inputs = processor(text=prompt, images=random_image, return_tensors="pt")

# Trace forward pass (NOT .generate!)
with model_addition_debugger_context(
    model,
    debug_path="./debug_output",
    do_prune_layers=False
):
    output = model.forward(**inputs)

# Files created:
# - debug_output/LlavaForConditionalGeneration_debug_tree_FULL_TENSORS.json
# - debug_output/LlavaForConditionalGeneration_debug_tree_SUMMARY.json
</syntaxhighlight>

=== Saving Full Tensors to SafeTensors ===
<syntaxhighlight lang="python">
from transformers import AutoModel, model_addition_debugger_context
import torch

model = AutoModel.from_pretrained("bert-base-uncased")
inputs = {
    "input_ids": torch.tensor([[101, 2023, 2003, 1037, 3231, 102]]),
    "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1]])
}

# Save full tensor data instead of repr
with model_addition_debugger_context(
    model,
    debug_path="./full_tensors",
    use_repr=False  # Saves .safetensors files
):
    outputs = model(**inputs)

# Creates JSON with paths like "./BertModel.encoder.layer.0_inputs.safetensors"
# Can load later with safetensors.torch.load_file()
</syntaxhighlight>

=== Keeping All Layers (No Pruning) ===
<syntaxhighlight lang="python">
from transformers import GPT2LMHeadModel, model_addition_debugger_context
import torch

model = GPT2LMHeadModel.from_pretrained("gpt2")
input_ids = torch.tensor([[464, 2415, 318, 1049]])

# Keep all layer traces (useful for detailed debugging)
with model_addition_debugger_context(
    model,
    debug_path="./detailed_trace",
    do_prune_layers=False  # Keep all intermediate layers
):
    outputs = model(input_ids)

# JSON will contain traces for all transformer blocks (0-11)
</syntaxhighlight>

=== Using in Distributed Training ===
<syntaxhighlight lang="python">
import torch
import torch.distributed as dist
from transformers import AutoModel, model_addition_debugger_context

# Initialize distributed training
dist.init_process_group(backend="nccl")

model = AutoModel.from_pretrained("roberta-base").to(
    torch.device(f"cuda:{dist.get_rank()}")
)

inputs = {...}  # Your distributed inputs

# Only rank 0 saves traces (handled automatically)
with model_addition_debugger_context(
    model,
    debug_path="./distributed_debug"
):
    outputs = model(**inputs)

# Only rank 0 creates JSON files
# Other ranks pass through without overhead
</syntaxhighlight>

=== Comparing Two Model Implementations ===
<syntaxhighlight lang="python">
from transformers import model_addition_debugger_context
import torch

# Original implementation
original_model = OriginalModel()
with model_addition_debugger_context(
    original_model,
    debug_path="./original",
    use_repr=True
):
    out1 = original_model(inputs)

# HuggingFace port
hf_model = HFModel()
with model_addition_debugger_context(
    hf_model,
    debug_path="./hf_port",
    use_repr=True
):
    out2 = hf_model(inputs)

# Now diff the SUMMARY.json files to find discrepancies
# Memory addresses are sanitized for clean diffs
</syntaxhighlight>

=== Minimal Example for Quick Check ===
<syntaxhighlight lang="python">
from transformers import AutoModel, model_addition_debugger_context
import torch

model = AutoModel.from_pretrained("distilbert-base-uncased")
inputs = torch.randint(0, 1000, (2, 10))

# Quick trace to current directory
with model_addition_debugger_context(model):
    outputs = model(input_ids=inputs)

# Creates DistilBertModel_debug_tree_*.json in current directory
</syntaxhighlight>

== Implementation Details ==

=== Forward Hook Mechanism ===
* Wraps every module's `forward` method using `functools.wraps`
* Maintains a call stack (`_debugger_model_call_stack`) to track nesting
* Parent modules collect children's trace nodes
* Original forward methods restored when context exits

=== No-Gradient Context ===
All tracing happens under `torch.no_grad()` to:
* Prevent memory overhead from autograd graph
* Ensure tensors can be safely detached and serialized
* Avoid interfering with actual training gradients

=== Distributed Handling ===
* Checks `torch.distributed.is_initialized()` and `get_rank()`
* Only rank 0 performs tracing and writes files
* Other ranks execute model normally with minimal overhead
* Supports DTensor objects (distributed tensors)

=== Memory Address Sanitization ===
Uses regex to replace patterns like `object at 0x7f8a4c12b0` with `object at 0xXXXXXXXX`:
* Enables clean diffs between traces
* Makes traces reproducible across runs
* Doesn't affect actual tensor values

=== Layer Pruning Logic ===
When `do_prune_layers=True`:
* Detects layer blocks by regex pattern `(.*)\.(\d+)$`
* Keeps first and last numbered layers
* Removes intermediate layers for readability
* Useful for large models (e.g., GPT-3 with 96 layers)

=== Output Pruning ===
* Leaf modules (no children) show outputs
* Parent modules (with children) have outputs removed
* Reduces JSON size and improves readability
* Outputs still visible in leaf nodes

=== Tensor Statistics ===
For float dtypes (fp16, fp32, bf16):
* Computes mean, std, min, max
* Helps identify distribution shifts
* Sanitized to remove memory addresses

=== File Naming ===
* Base name from model class name
* Path-based naming for SafeTensors (module path + inputs/outputs)
* Example: `BertModel.encoder.layer.0.attention.self_outputs.safetensors`

== Performance Considerations ==

=== Overhead ===
* Wrapping all modules adds call overhead
* Serialization is expensive for large models
* `use_repr=False` is faster but creates many files
* Intended for debugging, not production training

=== Disk Usage ===
* `use_repr=True`: 1 JSON file, potentially large (MB-GB range)
* `use_repr=False`: Many small SafeTensors files + JSON with paths
* SUMMARY.json typically 10-100x smaller than FULL_TENSORS.json

=== Memory Usage ===
* Keeps call stack in memory during forward pass
* Tensor serialization creates temporary CPU copies
* Large models may need significant RAM for tracing

== Related Pages ==
