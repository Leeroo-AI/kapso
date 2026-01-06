{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Model Loading Documentation|https://huggingface.co/docs/transformers/main_classes/model]]
* [[source::Doc|Accelerate Library|https://huggingface.co/docs/accelerate/]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Model_Management]], [[domain::Distributed_Computing]]
|-
! Last Updated
| [[last_updated::2025-12-18 00:00 GMT]]
|}

== Overview ==
Concrete tool for distributing model components across multiple devices using the Accelerate library, wrapped and integrated by HuggingFace Transformers.

=== Description ===
The dispatch_model function from Accelerate library is used by Transformers to implement sophisticated device placement strategies. Transformers integrates this external library to handle the complexity of splitting large models across GPUs, CPU, and disk. The integration includes automatic device map generation through the device_map="auto" parameter in from_pretrained(), which analyzes available GPU memory and automatically determines optimal layer placements. The Transformers wrapper handles offload directory management, special handling for tied weights, and integration with quantization strategies to ensure that device placement works correctly with quantized models.

This implementation leverages Accelerate's hooks system to transparently manage data movement between devices during forward passes, making multi-device execution seamless from the user's perspective.

=== Usage ===
Device dispatch via Accelerate is invoked automatically when loading models with device_map="auto" or a custom device map dictionary. It is used for models that exceed single GPU memory, for multi-GPU inference, or when implementing CPU/disk offloading for very large models on consumer hardware.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' src/transformers/integrations/accelerate.py (imports and usage)
* '''External Library:''' [https://github.com/huggingface/accelerate accelerate]
* '''Primary Function:''' accelerate.dispatch_model

=== Signature ===
<syntaxhighlight lang="python">
# From accelerate library, used by transformers
from accelerate import dispatch_model

def dispatch_model(
    model: nn.Module,
    device_map: dict,
    offload_dir: Optional[str] = None,
    offload_buffers: bool = False,
    skip_keys: Optional[list[str]] = None,
    preload_module_classes: Optional[list[str]] = None
) -> nn.Module
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM
# dispatch_model is used internally, not typically imported directly
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model || nn.Module || Yes || The PyTorch model to dispatch across devices
|-
| device_map || dict or str || Yes || Mapping of module names to devices, or "auto" for automatic generation
|-
| offload_dir || str || No || Directory path for offloading weights to disk
|-
| offload_buffers || bool || No || Whether to offload buffers in addition to parameters (default: False)
|-
| skip_keys || list[str] || No || Module names to skip during dispatch (keep on default device)
|-
| preload_module_classes || list[str] || No || Module classes to preload during offload (optimization)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| model || nn.Module || Dispatched model with layers assigned to specified devices and hooks registered
|}

== Usage Examples ==

=== Automatic Device Placement ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM
import torch

# Automatic device map generation (most common usage)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    device_map="auto",  # Automatically distribute across available GPUs
    torch_dtype=torch.float16
)

# Check device placement
for name, module in model.named_modules():
    if hasattr(module, 'weight') and module.weight is not None:
        print(f"{name}: {module.weight.device}")

# Example output:
# model.embed_tokens: cuda:0
# model.layers.0: cuda:0
# model.layers.15: cuda:1
# model.norm: cuda:1
# lm_head: cuda:1
</syntaxhighlight>

=== Manual Device Map ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM
import torch

# Custom device map for fine-grained control
device_map = {
    "model.embed_tokens": "cpu",
    "model.layers.0": "cuda:0",
    "model.layers.1": "cuda:0",
    "model.layers.2": "cuda:1",
    "model.layers.3": "cuda:1",
    "model.norm": "cuda:1",
    "lm_head": "cpu"
}

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    device_map=device_map,
    torch_dtype=torch.float16
)

print("Custom device placement applied")
</syntaxhighlight>

=== CPU Offloading for Large Models ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM
import torch

# Load very large model with CPU offloading
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    device_map="auto",
    offload_folder="./offload",  # Offload to disk if needed
    offload_state_dict=True,      # Offload during loading
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

# Some layers will be on disk, loaded on-demand during forward pass
print(f"Model loaded with offloading")
</syntaxhighlight>

=== Sequential Device Placement (Pipeline Parallelism) ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Place first half on GPU 0, second half on GPU 1
model = AutoModelForCausalLM.from_pretrained(
    "gpt2-large",
    device_map="balanced",  # Balanced split across available GPUs
    torch_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained("gpt2-large")

# Forward pass automatically handles cross-device transfers
inputs = tokenizer("Hello, world!", return_tensors="pt")
inputs = {k: v.to("cuda:0") for k, v in inputs.items()}  # Start on first device

outputs = model(**inputs)
print(f"Output device: {outputs.logits.device}")  # Will be on last device
</syntaxhighlight>

=== Checking Memory Usage Across Devices ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-13b-hf",
    device_map="auto",
    torch_dtype=torch.float16
)

# Check memory usage per device
device_memory = {}
for name, param in model.named_parameters():
    device = str(param.device)
    size = param.numel() * param.element_size()
    device_memory[device] = device_memory.get(device, 0) + size

print("Memory usage by device:")
for device, memory in device_memory.items():
    print(f"  {device}: {memory / 1e9:.2f} GB")
</syntaxhighlight>

=== Disk Offloading for Maximum Model Size ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM
import torch

# Load extremely large model with aggressive disk offloading
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    device_map="auto",
    offload_folder="./offload",
    offload_state_dict=True,
    torch_dtype=torch.float16,
    max_memory={
        0: "10GiB",  # Limit GPU 0 to 10GB
        1: "10GiB",  # Limit GPU 1 to 10GB
        "cpu": "30GiB"  # Limit CPU RAM to 30GB
    }
)

# Layers exceeding memory limits will be offloaded to disk
print("Model loaded with disk offloading enabled")
</syntaxhighlight>

=== Inspecting Device Map ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    device_map="auto",
    torch_dtype=torch.float16
)

# Access the device map
if hasattr(model, "hf_device_map"):
    print("Device Map:")
    for module_name, device in model.hf_device_map.items():
        print(f"  {module_name}: {device}")
else:
    print("No device map (model fits on single device)")
</syntaxhighlight>

=== Multi-GPU Inference with Balanced Load ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Balanced load across 2 GPUs
model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-6.7b",
    device_map="balanced",  # Equal split across GPUs
    torch_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-6.7b")

# Generate text (dispatch handles device transfers automatically)
prompt = "The future of artificial intelligence is"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Device_Placement]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_transformers_Loading_Environment]]

=== External Dependencies ===
* [[depends_on::Library:accelerate]] - HuggingFace Accelerate library for distributed training and inference
