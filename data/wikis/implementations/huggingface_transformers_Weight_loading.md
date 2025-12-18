{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Model Loading Documentation|https://huggingface.co/docs/transformers/main_classes/model]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Model_Management]]
|-
! Last Updated
| [[last_updated::2025-12-18 00:00 GMT]]
|}

== Overview ==
Concrete tool for deserializing checkpoint files into state dictionaries provided by HuggingFace Transformers.

=== Description ===
The load_state_dict function is the low-level utility that handles the actual reading and deserialization of model checkpoint files. It provides a unified interface for loading both safetensors and PyTorch pickle-based checkpoint files, with intelligent handling of different device targets and safety considerations. The function uses memory-mapped file access when possible for efficiency, supports meta-device loading for large models, and includes safety checks to prevent execution of malicious code in pickle files. It automatically detects the checkpoint format based on file extension and routes to the appropriate deserialization method.

This implementation is a critical component of the model loading pipeline, serving as the bridge between on-disk checkpoint files and in-memory tensor dictionaries.

=== Usage ===
load_state_dict is typically called internally during model loading by higher-level APIs like PreTrainedModel.from_pretrained(). Advanced users might call it directly when implementing custom weight loading strategies, debugging checkpoint issues, or inspecting model weights without loading the full model.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' src/transformers/modeling_utils.py
* '''Lines:''' 317-348

=== Signature ===
<syntaxhighlight lang="python">
def load_state_dict(
    checkpoint_file: Union[str, os.PathLike],
    map_location: Union[str, torch.device] = "cpu",
    weights_only: bool = True
) -> dict[str, torch.Tensor]
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from transformers.modeling_utils import load_state_dict
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| checkpoint_file || str or os.PathLike || Yes || Path to checkpoint file (.safetensors, .bin, .pt, or .pth)
|-
| map_location || str or torch.device || No || Device to load tensors onto: "cpu", "cuda:0", "meta", etc. (default: "cpu")
|-
| weights_only || bool || No || For PyTorch files, whether to restrict unpickling to tensors only (default: True for safety)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| state_dict || dict[str, torch.Tensor] || Dictionary mapping parameter names to tensor values
|}

== Usage Examples ==

=== Basic Checkpoint Loading ===
<syntaxhighlight lang="python">
from transformers.modeling_utils import load_state_dict

# Load safetensors checkpoint to CPU
state_dict = load_state_dict("model.safetensors")

print(f"Loaded {len(state_dict)} parameters")
for name, tensor in list(state_dict.items())[:3]:
    print(f"{name}: {tensor.shape}, {tensor.dtype}")

# Example output:
# embeddings.word_embeddings.weight: torch.Size([30522, 768]), torch.float32
# encoder.layer.0.attention.self.query.weight: torch.Size([768, 768]), torch.float32
</syntaxhighlight>

=== Loading to GPU ===
<syntaxhighlight lang="python">
from transformers.modeling_utils import load_state_dict
import torch

# Load directly to GPU
state_dict = load_state_dict(
    "model.safetensors",
    map_location="cuda:0"
)

print(f"All tensors on GPU: {state_dict['encoder.layer.0.attention.self.query.weight'].device}")
</syntaxhighlight>

=== Meta Device Loading for Large Models ===
<syntaxhighlight lang="python">
from transformers.modeling_utils import load_state_dict

# Load only metadata, no actual tensor data
state_dict = load_state_dict(
    "large_model.safetensors",
    map_location="meta"
)

# Tensors exist but have no allocated memory
for name, tensor in list(state_dict.items())[:3]:
    print(f"{name}: shape={tensor.shape}, device={tensor.device}, "
          f"dtype={tensor.dtype}, is_meta={tensor.is_meta}")

# Example output:
# embeddings.weight: shape=torch.Size([50257, 4096]), device=meta, dtype=torch.float32, is_meta=True
</syntaxhighlight>

=== Loading PyTorch Pickle Files ===
<syntaxhighlight lang="python">
from transformers.modeling_utils import load_state_dict

# Load PyTorch pickle checkpoint (with safety checks)
state_dict = load_state_dict(
    "pytorch_model.bin",
    map_location="cpu",
    weights_only=True  # Recommended for security
)

print(f"Loaded {len(state_dict)} parameters from PyTorch checkpoint")
</syntaxhighlight>

=== Comparing Checkpoint Formats ===
<syntaxhighlight lang="python">
from transformers.modeling_utils import load_state_dict
import time

# Load safetensors (fast, memory-mapped)
start = time.time()
state_dict_safe = load_state_dict("model.safetensors")
safe_time = time.time() - start

# Load PyTorch pickle (slower, full deserialization)
start = time.time()
state_dict_torch = load_state_dict("pytorch_model.bin")
torch_time = time.time() - start

print(f"Safetensors loading: {safe_time:.2f}s")
print(f"PyTorch loading: {torch_time:.2f}s")
print(f"Safetensors is {torch_time/safe_time:.1f}x faster")

# Verify they contain same weights
assert set(state_dict_safe.keys()) == set(state_dict_torch.keys())
</syntaxhighlight>

=== Inspecting Checkpoint Contents ===
<syntaxhighlight lang="python">
from transformers.modeling_utils import load_state_dict

# Load checkpoint to inspect without loading full model
state_dict = load_state_dict("model.safetensors")

# Analyze model size
total_params = sum(t.numel() for t in state_dict.values())
total_size_gb = sum(t.numel() * t.element_size() for t in state_dict.values()) / 1e9

print(f"Total parameters: {total_params:,}")
print(f"Total size: {total_size_gb:.2f} GB")

# Find specific layers
encoder_keys = [k for k in state_dict.keys() if 'encoder' in k]
print(f"Found {len(encoder_keys)} encoder parameters")

# Check data types
dtypes = set(t.dtype for t in state_dict.values())
print(f"Data types present: {dtypes}")
</syntaxhighlight>

=== Partial Loading for Debugging ===
<syntaxhighlight lang="python">
from transformers.modeling_utils import load_state_dict
import torch

# Load only metadata to check structure
state_dict = load_state_dict("large_model.safetensors", map_location="meta")

# Inspect structure without memory overhead
print("Model structure:")
for name in list(state_dict.keys())[:10]:
    print(f"  {name}")

# Later, load specific layers only (requires custom logic)
# This is useful for debugging specific model components
</syntaxhighlight>

=== Loading with Error Handling ===
<syntaxhighlight lang="python">
from transformers.modeling_utils import load_state_dict
import os

checkpoint_files = [
    "model.safetensors",
    "pytorch_model.bin",
    "model.bin"
]

state_dict = None
for checkpoint_file in checkpoint_files:
    if os.path.exists(checkpoint_file):
        try:
            state_dict = load_state_dict(checkpoint_file)
            print(f"Successfully loaded from {checkpoint_file}")
            break
        except Exception as e:
            print(f"Failed to load {checkpoint_file}: {e}")
            continue

if state_dict is None:
    raise FileNotFoundError("No valid checkpoint file found")
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_State_Dict_Loading]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_transformers_Loading_Environment]]
